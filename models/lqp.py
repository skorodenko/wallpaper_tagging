import torch
import lightning as lg
from .vcnn import VCNN
from .mlp import MLP
from torchvision.transforms import v2 as transforms


labels_f32 = transforms.Compose([
    transforms.Lambda(lambda x: x.sum(axis=1)),
    transforms.Lambda(lambda x: x.unsqueeze(1)),
])


class LQP(lg.LightningModule):
    
    def __init__(self, lr = 0.01, weight_decay = 0.01, models: dict = None):
        super().__init__()
        self.save_hyperparameters(ignore=["models"])
        if models:
            self.vcnn = models.get("vcnn", VCNN())
            self.mlp = models.get("mlp", MLP())
        self.fc = torch.nn.Sequential(
            torch.nn.Dropout(),
            torch.nn.Linear(81 * 2, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
        )
        self.loss_module = torch.nn.MSELoss()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr = self.hparams.lr,
            weight_decay = self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr = self.hparams.lr,
            epochs = self.trainer.max_epochs,
            steps_per_epoch = 3812,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
    
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
        f_labels = labels_f32(labels)
        loss = self.loss_module(pred, f_labels)
        self.log("train_loss", loss, on_step=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        (image, tags, labels) = batch
        pred = self.forward((image, tags))
        f_labels = labels_f32(labels)
        loss = self.loss_module(pred, f_labels)
        self.log("val_loss", loss, prog_bar=True)
    