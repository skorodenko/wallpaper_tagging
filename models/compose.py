import os
import torch
import lightning as lg
from torch import Tensor
from models.lp import LP
from models.mlp import MLP
from models.lqp import LQP
from models.vcnn import VCNN
from .utils import TagTransform, Metrics


label_transform = TagTransform("./assets/preprocessed/Labels_nus-wide.ndjson")


class Model(lg.LightningModule):
    
    def __init__(self, mode: str = "vcnn+mlp+lp+lqp", topk: int = 3):
        super().__init__()
        self.mode = mode
        self.topk = topk
        self.vcnn = VCNN()
        self.mlp = MLP()
        self.lp = LP()
        self.lqp = LQP()
        self.metrics = Metrics(81)
    
    def test_vcnn(self, x: Tensor, k: int = 3):
        image, _ = x
        pred = self.vcnn.predict(image)
        pred_topn = label_transform.decode_topn(pred, torch.tensor([k] * pred.shape[1]))
        return pred_topn

    def test_vcnn_mlp_lp(self, x: Tensor, k: int = 3):
        image, tags = x
        f_vis = self.vcnn.predict(image)
        f_text = self.mlp.predict(tags)
        f = torch.cat((f_vis, f_text), 1)
        pred = self.lp.predict(f)
        pred_topn = label_transform.decode_topn(pred, torch.tensor([k] * pred.shape[1]))
        return pred_topn
    
    def test_vcnn_lqp(self, x: Tensor):
        image, _ = x
        f_vis = self.vcnn.predict(image)
        f_text = torch.zeros_like(f_vis)
        f = torch.cat((f_vis, f_text), 1)
        pred = self.lp.predict(f)
        number = self.lqp.predict(f)
        number = number.round().to(torch.int64)
        pred_topn = label_transform.decode_topn(pred, number)
        return pred_topn
    
    def test_vcnn_mlp_lp_lqp(self, x: Tensor):
        image, tags = x
        f_vis = self.vcnn.predict(image)
        f_text = self.mlp.predict(tags)
        f = torch.cat((f_vis, f_text), 1)
        pred = self.lp.predict(f)
        number = self.lqp.predict(f)
        number = number.round().to(torch.int64)
        pred_topn = label_transform.decode_topn(pred, number)
        return pred_topn
    
    def forward(self, x):
        match self.mode:
            case "vcnn":
                return self.test_vcnn(x, k=self.topk)
            case "vcnn+lqp":
                return self.test_vcnn_lqp(x)
            case "vcnn+mlp+lp":
                return self.test_vcnn_mlp_lp(x, k=self.topk)
            case "vcnn+mlp+lp+lqp":
                return self.test_vcnn_mlp_lp_lqp(x)
    
    def predict_step(self, batch, batch_idx):
        image, tags, _ = batch
        pred = self.forward((image, tags))
        pred = pred.to(torch.int64)
        return label_transform.decode(pred)

    def test_step(self, batch, batch_idx):
        (image, tags, labels) = batch
        pred = self.forward((image, tags))
        pred = pred.to(torch.int64)
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