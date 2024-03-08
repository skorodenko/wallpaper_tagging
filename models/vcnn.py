import torch
import torch.nn as nn
import torchvision
import lightning as lg
from torch import Tensor
from models.utils import Metrics


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


resnet = torchvision.models.resnet101(
    weights=torchvision.models.ResNet101_Weights.IMAGENET1K_V2
)


class VCNN(lg.LightningModule):
    
    def __init__(self, lr: float = ..., weight_decay: float = ...):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
        )
        self.head = nn.Linear(2048, 1000)
        self.loss_module = nn.BCEWithLogitsLoss(
            pos_weight=torch.ones(1000),
            reduction="none",
        )
        self.activation = nn.Sigmoid()
        self.metrics = Metrics(1000)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr = self.hparams.lr,
            weight_decay = self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode = "min",
            factor = 0.75,
            patience = 2,
            threshold = 5.0,
            min_lr = 1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
        
    def predict(self, x: Tensor):
        x = self.forward(x)
        x = self.activation(x)
        return x
    
    def forward(self, x: Tensor):
        # Backbone forward
        x = self.backbone(x)
        
        # Output layer
        x = torch.flatten(x, 1)
        x = self.head(x)
        
        return x
    
    def training_step(self, batch, batch_idx):
        (image, _, labels) = batch
        pred = self.forward(image)
        loss = self.loss_module(pred, labels).max()
        self.log("train_loss", loss, on_step=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        (image, _, labels) = batch
        pred = self.forward(image)
        loss = self.loss_module(pred, labels).max()
        pred = self.activation(pred)
        pred = (pred > 0.45).to(torch.int64)
        labels = labels.to(torch.int64)
        self.metrics.update(pred, labels)
        self.log("val_loss", loss, prog_bar=True)
    
    def on_validation_epoch_end(self):
        cf1 = self.metrics.CF1()
        if1 = self.metrics.IF1()
        self.log("C-F1", cf1, prog_bar=True)
        self.log("I-F1", if1, prog_bar=True)
        self.metrics.reset()
    
    def test_step(self, batch, batch_idx):
        (image, _, labels) = batch
        pred = self.forward(image)
        pred = self.activation(pred)
        pred = (pred > 0.45).to(torch.int64)
        labels = labels.to(torch.int64)
        self.metrics.update(pred, labels)
    
    def on_test_epoch_end(self):
        cf1 = self.metrics.CF1()
        if1 = self.metrics.IF1()
        self.log("C-F1", cf1, prog_bar=True)
        self.log("I-F1", if1, prog_bar=True)
        self.metrics.reset()
