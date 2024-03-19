import torch
import lightning as lg
from .vcnn import VCNN
from .mlp import MLP
from .lp import LP
from .lqp import LQP
from pathlib import Path
from .utils import TagTransform, Metrics
from torchvision.transforms import v2 as transforms


MODEL_ROOT = Path("./assets/trained_models")
tag_transform = TagTransform("./assets/preprocessed/Labels_nus-wide.ndjson")


labels_f32 = transforms.Compose([
    transforms.Lambda(lambda x: x.sum(axis=1)),
    transforms.Lambda(lambda x: x.unsqueeze(1)),
])


class Model(lg.LightningModule):
    
    def __init__(self):
        super().__init__()
        self.mscnn = VCNN.load_from_checkpoint(MODEL_ROOT / "vcnn.train" / "vcnn.ckpt")
        self.mlp = MLP.load_from_checkpoint(MODEL_ROOT / "mlp.train" / "mlp.ckpt")
        self.lp = LP.load_from_checkpoint(MODEL_ROOT / "lp.train" / "lp.ckpt")
        self.lqp = LQP.load_from_checkpoint(MODEL_ROOT / "lqp.train" / "lqp.ckpt")
        self.metrics = Metrics(81)
        
    def forward(self, x):
        image, labels = x
        f_vis = self.mscnn.predict(image)
        f_text = self.mlp.predict(labels)
        f = torch.cat((f_vis, f_text), 1)
        labels = self.lp.predict(f)
        number = self.lqp.predict(f)
        number = number.round().to(torch.int64)
        labels_topn = tag_transform.decode_topn(labels, number)
        return labels_topn

    def test_step(self, batch, batch_idx):
        (image, tags, labels) = batch
        pred = self.forward((image, labels))
        pred = (pred > 0.5).to(torch.int64)
        tags = tags.to(torch.int64)
        self.metrics.update(pred, tags)
    
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