import torch
import lightning as lg
from .vcnn import VCNN
from .mlp import MLP
from models.utils import Metrics, TagTransform, nlabels_f32


label_transform = TagTransform("./assets/preprocessed/Labels_nus-wide.ndjson")


class LP(lg.LightningModule):
    
    def __init__(self, lr: float = ..., weight_decay: float = ..., models: dict = None):
        super().__init__()
        self.save_hyperparameters(ignore=["models"])
        if models:
            self.vcnn = models.get("vcnn", VCNN())
            self.mlp = models.get("mlp", MLP())
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(81 * 2, 81),
        )
        self.activation = torch.nn.Sigmoid()
        self.loss_module = torch.nn.BCEWithLogitsLoss()
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

    def predict(self, x):
        x = self.fc(x)
        x = self.activation(x)
        return x
        
    def forward(self, x):
        image, tags = x
        f_vis = self.vcnn.predict(image)
        f_text = self.mlp.predict(tags)
        f = torch.cat((f_vis, f_text), 1)
        x = self.fc(f)
        return x

    def training_step(self, batch, batch_idx):
        (image, tags, labels) = batch
        pred = self.forward((image, tags))
        loss = self.loss_module(pred, labels)
        self.log("train_loss", loss, on_step=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        (image, tags, labels) = batch
        pred = self.forward((image, tags))
        loss = self.loss_module(pred, labels)
        topn = nlabels_f32(labels)
        labels = labels.to(torch.int64)
        topn = topn.to(torch.int64)
        pred = label_transform.decode_topn(pred, topn)
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
    