from PIL import Image

import torch
from torch.utils.data import Dataset


class HappyWhaleDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df
        self.transform = transform

        self.image_names = self.df["image"].values
        self.image_paths = self.df["image_path"].values
        self.targets = self.df["individual_id"].values

    def __getitem__(self, index: int):
        image_name = self.image_names[index]

        image_path = self.image_paths[index]

        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        target = self.targets[index]
        target = torch.tensor(target, dtype=torch.long)

        return {"image_name": image_name, "image": image, "target": target}

    def __len__(self) -> int:
        return len(self.df)
