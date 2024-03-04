import torch
import lightning as lg
from .utils import TagEncoder
from .mscnn import MSCNN
from .mlp import MLP
from .lp import LP
from .lqp import LQP
from torchvision.transforms import v2 as transforms


labels_f32 = transforms.Compose([
    transforms.Lambda(lambda x: x.sum(axis=1)),
    transforms.Lambda(lambda x: x.unsqueeze(1)),
    transforms.Lambda(lambda x: x.divide(1063.0)),
])


tag_encoder = TagEncoder()


class Model(lg.LightningModule):
    
    def __init__(self):
        super().__init__()
        self.mscnn = MSCNN.load_from_checkpoint("./assets/trained_models/mscnn.train/mscnn.ckpt")
        self.mscnn.freeze()
        self.mlp = MLP.load_from_checkpoint("./assets/trained_models/mlp.train/mlp.ckpt")
        self.mlp.freeze()
        self.lp = LP.load_from_checkpoint("./assets/trained_models/lp.train/lp.ckpt")
        self.lp.freeze()
        self.lqp = LQP.load_from_checkpoint("./assets/trained_models/lqp.train/lqp.ckpt")
        self.lqp.freeze()
        
    def forward(self, x):
        image, tags = x
        f_vis = self.mscnn.predict(image)
        f_text = self.mlp.predict(tags)
        f = torch.cat((f_vis, f_text), 1)
        labels = self.lp.predict(f)
        number = self.lqp.predict(f)
        number = number.mul(1063).round().to(torch.int64)
        decoded = tag_encoder.decode(labels, number)
        return decoded

    def test_step(self, batch, batch_idx):
        (image, tags, labels) = batch
        pred = self.forward((image, tags))
        print(pred)