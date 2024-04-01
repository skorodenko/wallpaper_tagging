import torch
import lightning as lg
from .vcnn import VCNN
from .mlp import MLP
from .utils import nlabels_f32



class LQP(lg.LightningModule):
    
    def __init__(self, lr = 0.01, weight_decay = 0.01, models: dict = None):
        super().__init__()
        self.save_hyperparameters(ignore=["models"])
        if models:
            self.vcnn = models.get("vcnn", VCNN())
            self.mlp = models.get("mlp", MLP())
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(81 * 2, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, 1),
        )
        self.loss_module = torch.nn.MSELoss()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr = self.hparams.lr,
            weight_decay = self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[5,10], 
            gamma=0.5,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
    
    def predict(self, x):
        x = self.fc(x)
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
        f_labels = nlabels_f32(labels)
        loss = self.loss_module(pred, f_labels)
        self.log("train_loss", loss, on_step=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        (image, tags, labels) = batch
        pred = self.forward((image, tags))
        f_labels = nlabels_f32(labels)
        loss = self.loss_module(pred, f_labels)
        self.log("val_loss", loss, prog_bar=True)
    