import torch
import lightning as lg
from torch import Tensor
from models.utils import Metrics


class MLP(lg.LightningModule):
    
    def __init__(self, lr: float = ..., weight_decay: float = ...):
        super().__init__()
        self.save_hyperparameters()
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(1000, 2048),
            torch.nn.ReLU(inplace=True),
        )
        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(2048, 81),
        )
        self.activation = torch.nn.ReLU()  
        self.loss_module = torch.nn.BCEWithLogitsLoss()
        self.metrics = Metrics(81)
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr = self.hparams.lr,
            weight_decay = self.hparams.weight_decay,
            amsgrad = True,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr = self.hparams.lr,
            epochs = self.trainer.max_epochs,
            steps_per_epoch = 2490,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
    
    def predict(self, x: Tensor):
        x = self(x)
        x = self.activation(x)
        return x
    
    def forward(self, x: Tensor):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        (_, tags, labels) = batch
        pred = self.forward(labels)
        loss = self.loss_module(pred, tags)
        self.log("train_loss", loss, on_step=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        (_, tags, labels) = batch
        pred = self.forward(labels)
        loss = self.loss_module(pred, tags)
        pred = (self.activation(pred) > 0.5).to(torch.int64)
        tags = tags.to(torch.int64)
        self.metrics.update(pred, tags)
        self.log("val_loss", loss, prog_bar=True)
    
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
    
    def test_step(self, batch, batch_idx):
        (_, tags, labels) = batch
        pred = self.forward(labels)
        pred = (self.activation(pred) > 0.5).to(torch.int64)
        tags = tags.to(torch.int64)
        self.metrics.update(pred, tags)

    def on_test_epoch_end(self):
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
