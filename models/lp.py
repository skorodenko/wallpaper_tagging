import torch
import lightning as lg
from .vcnn import MSCNN
from .mlp import MLP


class LP(lg.LightningModule):
    
    def __init__(self, lr: float = ..., weight_decay: float = ..., momentum: float = ...):
        super().__init__()
        self.save_hyperparameters()
        self.mscnn = MSCNN.load_from_checkpoint("./assets/trained_models/mscnn.train/mscnn.ckpt")
        self.mscnn.freeze()
        self.mlp = MLP.load_from_checkpoint("./assets/trained_models/mlp.train/mlp.ckpt")
        self.mlp.freeze()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(1000 * 2, 1000),
        )
        self.activation = torch.nn.Softmax()
        self.loss_module = torch.nn.BCEWithLogitsLoss()
    
    def predict(self, x):
        x = self.fc(x)
        x = self.activation(x)
        return x
        
    def forward(self, x):
        image, tags = x
        f_vis = self.mscnn.predict(image)
        f_text = self.mlp.predict(tags)
        f = torch.cat((f_vis, f_text), 1)
        x = self.fc(f)
        return x

    def configure_optimizers(self):
        return torch.optim.SGD(
            self.parameters(),
            lr = self.hparams.lr,
            weight_decay = self.hparams.weight_decay,
            momentum = self.hparams.momentum,
        )

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
        self.log("val_loss", loss, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        (image, tags, labels) = batch
        pred = self.forward((image, tags))