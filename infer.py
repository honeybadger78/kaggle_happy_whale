import faiss
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from sklearn.preprocessing import LabelEncoder, normalize

from dataset.datamodule import LitDataModule
from utils.map_per_img import map_per_image

def load_eval_module(checkpoint_path):
    module= LitDataModule.load_from_checkpoint(checkpoint_path)
    module.to('cuda' if torch.cuda.is_available() else 'cpu')
    module.eval()
    
    return module

def load_dataloaders(
    train_csv_encoded_folded,
    test_csv,
    val_fold,
    image_size,
    batch_size,
    num_workers,
) :

    datamodule = LitDataModule(
        train_csv_encoded_folded=train_csv_encoded_folded,
        test_csv=test_csv,
        val_fold=val_fold,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    datamodule.setup()

    train_dl = datamodule.train_dataloader()
    val_dl = datamodule.val_dataloader()
    test_dl = datamodule.test_dataloader()

    return train_dl, val_dl, test_dl

def load_encoder():
    encoder = LabelEncoder()
    encoder.classes_ = np.load(ENCODER_CLASSES_PATH, allow_pickle=True)

    return encoder

@torch.inference_mode()
def get_embeddings(module, dataloader, encoder, stage):
    all_image_names= list()
    all_embeddings= list()
    all_targets= list()
    
    for batch in tqdm(dataloader, desc=f"Create {stage} Embedding"):
        image_names = batch["image_name"]
        images = batch["image"].to(module.device)
        targets = batch["target"].to(module.device)

        embeddings = module(images)

        all_image_names.append(image_names)
        all_embeddings.append(embeddings.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
        
    all_image_names = np.concatenate(all_image_names)
    all_embeddings = np.vstack(all_embeddings)
    all_targets = np.concatenate(all_targets)

    all_embeddings = normalize(all_embeddings, axis=1, norm="l2")
    all_targets = encoder.inverse_transform(all_targets)

    return all_image_names, all_embeddings, all_targets


def create_and_search_index(embedding_size, train_embeddings, val_embeddings, k):
    index= faiss.IndexFlatIP(embedding_size)
    index.add(train_embeddings)
    D, I= index.search(val_embeddings, k)
    
    return D, I

def create_val_targets_df(
    train_targets, val_image_names, val_targets
):

    allowed_targets = np.unique(train_targets)
    val_targets_df = pd.DataFrame(np.stack([val_image_names, val_targets], axis=1), columns=["image", "target"])
    val_targets_df.loc[~val_targets_df.target.isin(allowed_targets), "target"] = "new_individual"

    return val_targets_df


def create_distances_df(
    image_names, targets, D, I, stage 
):

    distances_df = []
    for i, image_name in tqdm(enumerate(image_names), desc=f"Creating {stage}_df"):
        target = targets[I[i]]
        distances = D[i]
        subset_preds = pd.DataFrame(np.stack([target, distances], axis=1), columns=["target", "distances"])
        subset_preds["image"] = image_name
        distances_df.append(subset_preds)

    distances_df = pd.concat(distances_df).reset_index(drop=True)
    distances_df = distances_df.groupby(["image", "target"]).distances.max().reset_index()
    distances_df = distances_df.sort_values("distances", ascending=False).reset_index(drop=True)

    return distances_df


def get_best_threshold(val_targets_df, valid_df) :
    best_th = 0
    best_cv = 0
    for th in [0.1 * x for x in range(11)]:
        all_preds = get_predictions(valid_df, threshold=th)

        cv = 0
        for i, row in val_targets_df.iterrows():
            target = row.target
            preds = all_preds[row.image]
            val_targets_df.loc[i, th] = map_per_image(target, preds)

        cv = val_targets_df[th].mean()

        print(f"th={th} cv={cv}")

        if cv > best_cv:
            best_th = th
            best_cv = cv

    print(f"best_th={best_th}")
    print(f"best_cv={best_cv}")

    # Adjustment: Since Public lb has nearly 10% 'new_individual' (Be Careful for private LB)
    val_targets_df["is_new_individual"] = val_targets_df.target == "new_individual"
    val_scores = val_targets_df.groupby("is_new_individual").mean().T
    val_scores["adjusted_cv"] = val_scores[True] * 0.1 + val_scores[False] * 0.9
    best_th = val_scores["adjusted_cv"].idxmax()
    print(f"best_th_adjusted={best_th}")

    return best_th, best_cv

def get_predictions(df, threshold):
    sample_list = ["938b7e931166", "5bf17305f073", "7593d2aee842", "7362d7a01d00", "956562ff2888"]

    predictions = {}
    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Creating predictions for threshold={threshold}"):
        if row.image in predictions:
            if len(predictions[row.image]) == 5:
                continue
            predictions[row.image].append(row.target)
        elif row.distances > threshold:
            predictions[row.image] = [row.target, "new_individual"]
        else:
            predictions[row.image] = ["new_individual", row.target]

    for x in tqdm(predictions):
        if len(predictions[x]) < 5:
            remaining = [y for y in sample_list if y not in predictions]
            predictions[x] = predictions[x] + remaining
            predictions[x] = predictions[x][:5]

    return predictions

def create_predictions_df(test_df, best_th):
    predictions = get_predictions(test_df, best_th)

    predictions = pd.Series(predictions).reset_index()
    predictions.columns = ["image", "predictions"]
    predictions["predictions"] = predictions["predictions"].apply(lambda x: " ".join(x))

    return predictions


def infer(
    checkpoint_path: str,
    train_csv_encoded_folded: str = str(TRAIN_CSV_ENCODED_FOLDED_PATH),
    test_csv: str = str(TEST_CSV_PATH),
    val_fold: float = 0.0,
    image_size: int = 256,
    batch_size: int = 64,
    num_workers: int = 2,
    k: int = 50,
):
    module = load_eval_module(checkpoint_path, torch.device("cuda"))

    train_dl, val_dl, test_dl = load_dataloaders(
        train_csv_encoded_folded=train_csv_encoded_folded,
        test_csv=test_csv,
        val_fold=val_fold,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    encoder = load_encoder()

    train_image_names, train_embeddings, train_targets = get_embeddings(module, train_dl, encoder, stage="train")
    val_image_names, val_embeddings, val_targets = get_embeddings(module, val_dl, encoder, stage="val")
    test_image_names, test_embeddings, test_targets = get_embeddings(module, test_dl, encoder, stage="test")

    D, I = create_and_search_index(module.hparams.embedding_size, train_embeddings, val_embeddings, k)  # noqa: E741
    print("Created index with train_embeddings")

    val_targets_df = create_val_targets_df(train_targets, val_image_names, val_targets)
    print(f"val_targets_df=\n{val_targets_df.head()}")

    val_df = create_distances_df(val_image_names, train_targets, D, I, "val")
    print(f"val_df=\n{val_df.head()}")

    best_th, best_cv = get_best_threshold(val_targets_df, val_df)
    print(f"val_targets_df=\n{val_targets_df.describe()}")

    train_embeddings = np.concatenate([train_embeddings, val_embeddings])
    train_targets = np.concatenate([train_targets, val_targets])
    print("Updated train_embeddings and train_targets with val data")

    D, I = create_and_search_index(module.hparams.embedding_size, train_embeddings, test_embeddings, k)  # noqa: E741
    print("Created index with train_embeddings")

    test_df = create_distances_df(test_image_names, train_targets, D, I, "test")
    print(f"test_df=\n{test_df.head()}")

    predictions = create_predictions_df(test_df, best_th)
    print(f"predictions.head()={predictions.head()}")
    

    public_predictions = pd.read_csv(PUBLIC_SUBMISSION_CSV_PATH)
    ids_without_backfin = np.load(IDS_WITHOUT_BACKFIN_PATH, allow_pickle=True)

    ids2 = public_predictions["image"][~public_predictions["image"].isin(predictions["image"])]

    predictions = pd.concat(
        [
            predictions[~(predictions["image"].isin(ids_without_backfin))],
            public_predictions[public_predictions["image"].isin(ids_without_backfin)],
            public_predictions[public_predictions["image"].isin(ids2)],
        ]
    )
    predictions = predictions.drop_duplicates()

    predictions.to_csv(SUBMISSION_CSV_PATH, index=False)