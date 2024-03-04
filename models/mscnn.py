import torch
import torch.nn as nn
import torchvision
import lightning as lg
from torch import Tensor
from torcheval.metrics.functional import multiclass_precision, multiclass_recall, multiclass_f1_score
from torcheval.metrics import MulticlassPrecision


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


resnet = torchvision.models.resnet101(
    weights=torchvision.models.ResNet101_Weights.IMAGENET1K_V2
)

class FusionBlock(nn.Module):
    
    def __init__(self, in_planes: int, out_planes: int, main_layer: int, stride: int = 1):
        super().__init__()
        self.main_layer = getattr(resnet, f"layer{main_layer}")
        self.conv1 = nn.Sequential(
            conv3x3(in_planes, in_planes, stride),
            nn.BatchNorm2d(in_planes),
            nn.ReLU() 
        )
        self.conv2 = nn.Sequential(
            conv1x1(in_planes, out_planes),
            nn.BatchNorm2d(out_planes),
            nn.ReLU()
        )
        
    def forward(self, x: Tensor):
        out = self.main_layer(x)
        fusion = self.conv1(x)
        fusion = self.conv2(fusion)
        out = out + fusion
        return out


class MSCNN(lg.LightningModule):
    
    def __init__(self, lr: float = ..., weight_decay: float = ..., momentum: float = ...):
        super().__init__()
        self.save_hyperparameters()
        self.conv1 = resnet.conv1
        self.fb1 = FusionBlock(64, 256, 1)
        self.fb2 = FusionBlock(256, 512, 2, 2)
        self.fb3 = FusionBlock(512, 1024, 3, 2)
        self.fb4 = FusionBlock(1024, 2048, 4, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(2048, 1063)
        self.loss_module = nn.BCEWithLogitsLoss()
        self.activation = nn.Sigmoid()
        self.c_prec = MulticlassPrecision(num_classes=1063, average="macro")
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr = self.hparams.lr,
            weight_decay = self.hparams.weight_decay,
            momentum = self.hparams.momentum,
        )
        return optimizer
        
    def predict(self, x: Tensor):
        x = self.forward(x)
        x = self.activation(x)
        return x
    
    def forward(self, x: Tensor):
        # Layer 1
        x = self.conv1(x)
        
        # Layers 2-5
        x = self.fb1(x)
        x = self.fb2(x)
        x = self.fb3(x)
        x = self.fb4(x)
        
        # Output layer
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
    def training_step(self, batch, batch_idx):
        (image, _, labels) = batch
        pred = self.forward(image)
        loss = self.loss_module(pred, labels)
        self.log("train_loss", loss, on_step=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        (image, _, labels) = batch
        pred = self.predict(image)
        loss = self.loss_module(pred, labels)
        print(labels.shape, pred.shape)
        _labels = torch.argmax(labels, dim=1)
        print(_labels)
        print(pred)
        self.c_prec.update(pred, _labels)
#        c_prec = multiclass_precision(pred, labels, num_classes=1063, average="macro")
#        c_recall = multiclass_recall(pred, labels, num_classes=1063, average="macro")
#        c_f1 = multiclass_f1_score(pred, labels, num_classes=1063, average="macro")
#        i_prec = multiclass_precision(pred, labels, num_classes=1063, average="micro")
#        i_recall = multiclass_recall(pred, labels, num_classes=1063, average="micro")
#        i_f1 = multiclass_f1_score(pred, labels, num_classes=1063, average="micro")
        self.log("val_loss", loss, on_epoch=True)
    
    def on_validation_epoch_end(self):
        self.log("c_prec", self.c_prec.compute())
#        self.log("c_recall", c_recall, prog_bar=True, on_epoch=True)
#        self.log("c_f1", c_f1, prog_bar=True, on_epoch=True)
#        self.log("i_prec", i_prec, on_epoch=True)
#        self.log("i_recall", i_recall, on_epoch=True)
#        self.log("i_f1", i_f1, prog_bar=True, on_epoch=True)
        self.c_prec.reset()
    
    def test_step(self, batch, batch_idx):
        (image, _, labels) = batch
        pred = self.forward(image)