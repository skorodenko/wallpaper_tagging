import torch
import lightning as lg
from torch import Tensor
from models.utils import Metrics, TagTransform


label_transform = TagTransform("./assets/preprocessed/Labels_nus-wide.ndjson")


class MLP(lg.LightningModule):
    
    def __init__(self, lr: float = ..., weight_decay: float = ...):
        super().__init__()
        self.save_hyperparameters()
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(1000, 2048),
            torch.nn.ReLU(),
        )
        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(2048, 81),
        )
        self.activation = torch.nn.Sigmoid()  
        self.loss_module = torch.nn.BCEWithLogitsLoss(
            pos_weight = torch.ones(81) * 2
        )
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
            gamma=0.1
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
    
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
        pred = self.forward(tags)
        loss = self.loss_module(pred, labels)
        self.log("train_loss", loss, on_step=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        (_, tags, labels) = batch
        pred = self.forward(tags)
        loss = self.loss_module(pred, labels)
        labels = labels.to(torch.int64)
        pred = label_transform.decode_topn(pred, torch.tensor([3] * pred.shape[1]))
        pred = pred.to(torch.int64)
        self.metrics.update(pred, labels)
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
    