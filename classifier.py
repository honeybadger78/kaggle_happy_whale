import timm
from timm.optim import create_optimizer_v2

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from utils.arcface import ArcMarginProduct


class Classifier(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        pretrained: bool,
        drop_rate: float,
        embedding_size: int,
        num_classes: int,
        arc_s: float,
        arc_m: float,
        arc_easy_margin: bool,
        arc_ls_eps: float,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.model = timm.create_model(
            model_name, pretrained=pretrained, drop_rate=drop_rate
        )
        self.embedding = nn.Linear(
            self.model.get_classifier().in_features, embedding_size
        )
        self.model.reset_classifier(num_classes=0, global_pool="avg")

        self.arc = ArcMarginProduct(
            in_features=embedding_size,
            out_features=num_classes,
            s=arc_s,
            m=arc_m,
            easy_margin=arc_easy_margin,
            ls_eps=arc_ls_eps,
        )

        self.loss_fn = F.cross_entropy

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.model(images)
        embeddings = self.embedding(features)

        return embeddings

    def configure_optimizers(self):
        optimizer = create_optimizer_v2(
            self.parameters(),
            opt=self.hparams.optimizer,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            self.hparams.learning_rate,
            steps_per_epoch=self.hparams.len_train_dl,
            epochs=self.hparams.epochs,
        )
        scheduler = {"scheduler": scheduler, "interval": "step"}

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        return self._step(batch, "val")

    def _step(self, batch, step) -> torch.Tensor:
        images, targets = batch["image"], batch["target"]

        embeddings = self(images)
        outputs = self.arc(embeddings, targets, self.device)

        loss = self.loss_fn(outputs, targets)

        self.log(f"{step}_loss", loss)

        return loss
