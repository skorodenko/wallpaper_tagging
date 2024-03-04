import torch
import lightning as lg
from .mscnn import MSCNN
from .mlp import MLP
from torchvision.transforms import v2 as transforms


labels_f32 = transforms.Compose([
    transforms.Lambda(lambda x: x.sum(axis=1)),
    transforms.Lambda(lambda x: x.unsqueeze(1)),
    transforms.Lambda(lambda x: x.divide(1063.0)),
])


class LQP(lg.LightningModule):
    
    def __init__(self, lr = 0.001, weight_decay = 0.0001):
        super().__init__()
        self.save_hyperparameters()
        self.mscnn = MSCNN.load_from_checkpoint("./assets/trained_models/mscnn.train/mscnn.ckpt")
        self.mscnn.freeze()
        self.mlp = MLP.load_from_checkpoint("./assets/trained_models/mlp.train/mlp.ckpt")
        self.mlp.freeze()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(1063 * 2, 512),
            torch.nn.Dropout(),
            torch.nn.Linear(512, 256),
            torch.nn.Dropout(),
            torch.nn.Linear(256, 1),
        )
        self.loss_module = torch.nn.MSELoss()

    def predict(self, x):
        x = self.fc(x)
        return x
    
    def forward(self, x):
        image, tags = x
        f_vis = self.mscnn.predict(image)
        f_text = self.mlp.predict(tags)
        f = torch.cat((f_vis, f_text), 1)
        x = self.fc(f)
        return x

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr = self.hparams.lr,
            weight_decay = self.hparams.weight_decay,
        )

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
    
    def test_step(self, batch, batch_idx):
        (image, tags, labels) = batch
        pred = self.forward((image, tags))