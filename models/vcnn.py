import torch
import torch.nn as nn
import torchvision
import lightning as lg
from torch import Tensor
from models.utils import Metrics


resnet = torchvision.models.resnet101(
    weights=torchvision.models.ResNet101_Weights.IMAGENET1K_V2
)


class VCNN(lg.LightningModule):
    
    def __init__(self, lr: float = ..., weight_decay: float = ...):
        super().__init__()
        self.save_hyperparameters()
        self.conv0 = resnet.conv1
        self.bn0 = resnet.bn1
        self.relu0 = resnet.relu
        self.maxpool0 = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self.head = nn.Linear(2048, 1000)
        self.loss_module = nn.BCEWithLogitsLoss(
            reduction = "sum",
        )
        self.activation = nn.Sigmoid()
        self.metrics = Metrics(1000)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr = self.hparams.lr,
            weight_decay = self.hparams.weight_decay,
            amsgrad = True,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones = [5,10],
            gamma = 0.1,
        )
        return [optimizer], [scheduler]
        
    def predict(self, x: Tensor):
        x = self.forward(x)
        x = self.activation(x)
        return x
    
    def forward(self, x: Tensor):
        # Resnet
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)
        x = self.maxpool0(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        
        # Output layer
        x = torch.flatten(x, 1)
        x = self.head(x)
        
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
    
    def on_validation_epoch_end(self):
        cf1 = self.metrics.CF1()
        if1 = self.metrics.IF1()
        self.log("C_F1", cf1, prog_bar=True)
        self.log("I_F1", if1, prog_bar=True)
        self.metrics.reset()
    
    def test_step(self, batch, batch_idx):
        (image, _, labels) = batch
        pred = self.forward(image)
        pred = (self.activation(pred) > 0.5).to(torch.int64)
        labels = labels.to(torch.int64)
        self.metrics.update(pred, labels)
    
    def on_test_epoch_end(self):
        cf1 = self.metrics.CF1()
        if1 = self.metrics.IF1()
        self.log("C_F1", cf1, prog_bar=True)
        self.log("I_F1", if1, prog_bar=True)
        self.metrics.reset()
