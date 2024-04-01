import torch
import torch.nn as nn
import torchvision
import lightning as lg
from torch import Tensor
from models.utils import Metrics


class VCNN(lg.LightningModule):
    
    def __init__(self, lr: float = ..., weight_decay: float = ...):
        super().__init__()
        self.save_hyperparameters()
        base = torchvision.models.resnext101_32x8d(
            weights=torchvision.models.ResNeXt101_32X8D_Weights.IMAGENET1K_V2
        )
        self.nfilters = base.fc.in_features
        self.feature_extractor = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
            base.avgpool,
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Linear(self.nfilters, 81),
        )
        self.loss_module = nn.BCEWithLogitsLoss(
            pos_weight = torch.ones(81) * 2
        )
        self.activation = nn.Sigmoid()
        self.metrics = Metrics(81)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr = self.hparams.lr,
            weight_decay = self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[5,10], 
            gamma=0.5
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
        
    def predict(self, x: Tensor):
        x = self.forward(x)
        x = self.activation(x)
        return x
    
    def forward(self, x: Tensor):
        x = self.feature_extractor(x)
        x = self.fc(x)
        return x
    
    def training_step(self, batch, batch_idx):
        (image, _, labels) = batch
        pred = self.forward(image)
        loss = self.loss_module(pred, labels)
        self.log("train_loss", loss, on_step=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        (image, _, labels) = batch
        pred = self.forward(image)
        loss = self.loss_module(pred, labels)
        pred = (self.activation(pred) > 0.5).to(torch.int64)
        labels = labels.to(torch.int64)
        self.metrics.update(pred, labels)
        self.log("val_loss", loss, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)
    
    def on_validation_epoch_end(self):
        cp, cr = self.metrics.CP(), self.metrics.CR()
        cf1 = self.metrics.CF1()
        ip, ir = self.metrics.IP(), self.metrics.IR()
        if1 = self.metrics.IF1()
        hf1 = self.metrics.HF1()
        self.log("CP", cp)
        self.log("CR", cr)
        self.log("IP", ip)
        self.log("IR", ir)
        self.log("C_F1", cf1)
        self.log("I_F1", if1)
        self.log("H_F1", hf1, prog_bar=True)
        self.metrics.reset()
    
    def on_test_epoch_end(self):
        self.on_validation_epoch_end()