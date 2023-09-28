import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from configs.config import *

def get_image_path(id: str, dir) -> str:
    return f"{dir / id}"


def prepare_train_data(configs):
    train_df = pd.read_csv(TRAIN_CSV_PATH)

    train_df["image_path"] = train_df["image"].apply(
        get_image_path, dir=TRAIN_DIR
    )

    encoder = LabelEncoder()
    train_df["individual_id"] = encoder.fit_transform(train_df["individual_id"])
    np.save(ENCODER_CLASSES_PATH, encoder.classes_)

    skf = StratifiedKFold(n_splits=N_SPLITS)
    for fold, (_, val_) in enumerate(skf.split(X=train_df, y=train_df.individual_id)):
        train_df.loc[val_, "kfold"] = fold

    train_df.to_csv(TRAIN_CSV_ENCODED_FOLDED_PATH, index=False)


def prepare_test_data(configs):
    test_df = pd.read_csv(SAMPLE_SUBMISSION_CSV_PATH)
    test_df["image_path"] = test_df["image"].apply(
        get_image_path, dir=TEST_DIR
    )
    test_df.drop(columns=["predictions"], inplace=True)
    # Dummy id
    test_df["individual_id"] = 0
    test_df.to_csv(TEST_CSV_PATH, index=False)
