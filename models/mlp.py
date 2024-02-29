import torch
import lightning as lg


class MLP(lg.LightningModule):
    
    def __init__(self, lr = 0.001, weight_decay = 0.00001):
        super().__init__()
        self.save_hyperparameters()
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(1063, 2048),
            torch.nn.ReLU(),
        )
        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(2048, 1063),
            torch.nn.ReLU(),  
        )
        self.loss_module = torch.nn.CrossEntropyLoss()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr = self.hparams.lr,
            weight_decay = self.hparams.weight_decay,
        )

    def training_step(self, batch, batch_idx):
        (_, tags, labels) = batch
        pred = self.forward(tags)
        loss = self.loss_module(pred, labels)
        #acc = (pred.argmax(dim=-1) == labels).float().mean()
        #self.log("train_acc", acc, on_step=False, on_epoch=True)
        self.log("train_loss", loss, on_step=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        (_, tags, labels) = batch
        pred = self.forward(tags)
        loss = self.loss_module(pred, labels)
        #acc = (pred.argmax(dim=-1) == labels).float().mean()
        #self.log("val_acc", acc, on_step=False, on_epoch=True)
        self.log("val_loss", loss, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        (_, tags, labels) = batch
        pred = self.forward(tags)
        #acc = (pred.argmax(dim=-1) == labels).float().mean()
        #self.log("test_acc", acc, on_step=False, on_epoch=True)