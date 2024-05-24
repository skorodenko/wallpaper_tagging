import torch
import torchvision
import lightning as lg
from torch import Tensor
from argparse import Namespace
from .utils import TagTransform, Metrics


label_transform = TagTransform("./assets/preprocessed/Labels_nus-wide.ndjson")


class TagAttention(lg.LightningModule):
    
    def __init__(self, params: Namespace = None, mode: str = "image+tags"):
        super().__init__()
        self.save_hyperparameters(params)
        self.mode = mode
        self.metrics = Metrics(81)
        base = torchvision.models.resnet101(
            weights=torchvision.models.ResNet101_Weights.IMAGENET1K_V2
        )
        self.vis_transform = torch.nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
            base.avgpool,
            torch.nn.Flatten(),
            torch.nn.Linear(base.fc.in_features, 512)
        )
        self.tags_transform = torch.nn.Sequential(
            torch.nn.Linear(1000, 2048),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(2048, 2048),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(2048, 512)
        )
        self.attention = torch.nn.MultiheadAttention(
            embed_dim=512, num_heads=64,
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512, 81),
        )
        self.loss_module = torch.nn.BCEWithLogitsLoss()
        self.activation = torch.nn.Sigmoid()
        self.metrics = Metrics(81)
    
    def test_image(self, x: Tensor, k: int = 3):
        image, _ = x
        image_embed = self.vis_transform(image)
        tags_embed = torch.zeros_like(image_embed)
        embed = torch.cat((image_embed, tags_embed), dim = 1)
        batch_size = image_embed.shape[0]
        embed = embed.reshape((batch_size, 2, 512))
        pred, _ = self.attention(embed, embed, embed, need_weights=False)
        pred = pred.sum(dim = 1)
        pred = self.classifier(pred)
        return pred

    def test_image_tags(self, x: Tensor, k: int = 3):
        image, tags = x
        image_embed = self.vis_transform(image)
        tags_embed = self.tags_transform(tags)
        embed = torch.cat((image_embed, tags_embed), dim = 1)
        batch_size = image_embed.shape[0]
        embed = embed.reshape((batch_size, 2, 512))
        pred, _ = self.attention(embed, embed, embed, need_weights=False)
        pred = pred.sum(dim = 1)
        pred = self.classifier(pred)
        return pred
    
    def forward(self, x):
        match self.mode:
            case "image":
                return self.test_image(x)
            case "image+tags":
                return self.test_image_tags(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr = self.hparams.lr,
            weight_decay = self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[5,10], 
            gamma=self.hparams.sched_gamma,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
    
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
        labels = labels.to(torch.int64)
        pred = self.activation(pred)
        pred = label_transform.decode_threshold(pred, 0.5)
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
    
    def predict_step(self, batch, batch_idx):
        (image, tags, labels) = batch
        pred = self.forward((image, tags))
        labels = labels.to(torch.int64)
        pred = self.activation(pred)
        pred = label_transform.decode_threshold(pred, 0.5)
        pred = pred.to(torch.int64)
        return label_transform.decode(pred)
    
    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)
    
    def on_test_epoch_end(self):
        self.on_validation_epoch_end()