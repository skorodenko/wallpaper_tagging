import torch
import torch.nn as nn
import torchvision
import lightning as lg
from torch import Tensor


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
        #if main_layer in [1,2,3,4]:
        #    self.main_layer.requires_grad_(False)
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
    
    def __init__(self, lr = 0.001, weight_decay = 0.00001):
        super().__init__()
        self.save_hyperparameters()
        self.conv1 = resnet.conv1
        #self.conv1.requires_grad_(False)
        self.fb1 = FusionBlock(64, 256, 1)
        self.fb2 = FusionBlock(256, 512, 2, 2)
        self.fb3 = FusionBlock(512, 1024, 3, 2)
        self.fb4 = FusionBlock(1024, 2048, 4, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(2048, 1063)
        self.loss_module = nn.CrossEntropyLoss()
    
    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr = self.hparams.lr,
            weight_decay = self.hparams.weight_decay,
        )
    
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
        #acc = (pred.argmax(dim=-1) == labels).float().mean()
        #self.log("train_acc", acc, on_step=False, on_epoch=True)
        self.log("train_loss", loss, on_step=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        (image, _, labels) = batch
        pred = self.forward(image)
        loss = self.loss_module(pred, labels)
        #acc = (pred.argmax(dim=-1) == labels).float().mean()
        #self.log("val_acc", acc, on_step=False, on_epoch=True)
        self.log("val_loss", loss)
    
    def test_step(self, batch, batch_idx):
        (image, _, labels) = batch
        pred = self.forward(image)
        #acc = (pred.argmax(dim=-1) == labels).float().mean()
        #self.log("test_acc", acc, on_step=False, on_epoch=True)