import torch
import lightning as lg
from models.lp import LP
from models.mlp import MLP
from models.lqp import LQP
from models.vcnn import VCNN
from .utils import TagTransform, Metrics
from torchvision.transforms import v2 as transforms


tag_transform = TagTransform("./assets/preprocessed/Labels_nus-wide.ndjson")


labels_f32 = transforms.Compose([
    transforms.Lambda(lambda x: x.sum(axis=1)),
    transforms.Lambda(lambda x: x.unsqueeze(1)),
])


class Model(lg.LightningModule):
    
    def __init__(self, models: dict = None):
        super().__init__()
        if models:
            self.vcnn = models["vcnn"]
            self.mlp = models["mlp"]
            self.lp = models["lp"]
            self.lqp = models["lqp"]
        else:
            self.vcnn = VCNN()
            self.mlp = MLP()
            self.lp = LP()
            self.lqp = LQP()
        self.metrics = Metrics(81)
        
    def forward(self, x):
        image, tags = x
        f_vis = self.vcnn.predict(image)
        f_text = self.mlp.predict(tags)
        f = torch.cat((f_vis, f_text), 1)
        pred = self.lp.predict(f)
        number = self.lqp.predict(f)
        number = number.round().to(torch.int64)
        pred_topn = tag_transform.decode_topn(pred, number)
        return pred_topn
    
    def predict_step(self, batch, batch_idx):
        (image, tags, labels) = batch
        pred = self.forward((image, tags))
        pred = (pred > 0.5).to(torch.int64)
        return tag_transform.decode(pred)

    def test_step(self, batch, batch_idx):
        (image, tags, labels) = batch
        pred = self.forward((image, tags))
        pred = (pred > 0.5).to(torch.int64)
        labels = labels.to(torch.int64)
        self.metrics.update(pred, labels)
    
    def on_test_epoch_end(self):
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