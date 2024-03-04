import torch
import lightning as lg
from torch import Tensor


class MLP(lg.LightningModule):
    
    def __init__(self, lr: float = ..., weight_decay: float = ..., momentum: float = ...):
        super().__init__()
        self.save_hyperparameters()
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(1063, 2048),
            torch.nn.ReLU(),
        )
        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(2048, 1063),
        )
        self.activation = torch.nn.ReLU()  
        self.loss_module = torch.nn.BCEWithLogitsLoss()
        
    def predict(self, x: Tensor):
        x = self(x)
        x = self.activation(x)
        return x
    
    def forward(self, x: Tensor):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def configure_optimizers(self):
        return torch.optim.SGD(
            self.parameters(),
            lr = self.hparams.lr,
            weight_decay = self.hparams.weight_decay,
            momentum = self.hparams.momentum,
        )

    def training_step(self, batch, batch_idx):
        (_, tags, labels) = batch
        pred = self.forward(tags)
        loss = self.loss_module(pred, labels)
        self.log("train_loss", loss, on_step=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        (_, tags, labels) = batch
        pred = self.forward(tags)
        loss = self.loss_module(pred, labels)
        self.log("val_loss", loss, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        (_, tags, labels) = batch
        pred = self.forward(tags)