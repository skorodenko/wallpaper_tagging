import torch
import numpy as np
import polars as pl
import torch.nn.functional as F
from torch import Tensor
from pathlib import Path
from typing import Iterable
from collections import defaultdict
from torchvision.transforms import v2 as transforms
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from lightning.pytorch.callbacks import BaseFinetuning


nlabels_f32 = transforms.Compose([
    transforms.Lambda(lambda x: x.sum(axis=1)),
    transforms.Lambda(lambda x: x.unsqueeze(1)),
])

# Calculated in eda.ipynb
LAB_MEAN = 2.4186057952477
LAB_STD = 1.5880828167376047

nlabels_normalize = transforms.Compose([
    transforms.Lambda(lambda x: (x - LAB_MEAN) / LAB_STD)
])

nlabels_denormalize = transforms.Compose([
    transforms.Lambda(lambda x: (x * LAB_STD) + LAB_MEAN)
])


class Metrics:
    
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()
    
    @property
    def cls_correct(self):
        return self._cls_correct
    
    @property
    def cls_pred(self):
        return self._cls_pred
    
    @property
    def cls_ground(self):
        return self._cls_ground
    
    @property
    def im_correct(self):
        return self._im_correct
    
    @property
    def im_pred(self):
        return self._im_pred
    
    @property
    def im_ground(self):
        return self._im_ground
    
    def update(self, pred: Tensor, target: Tensor):
        cls_correct = ((pred == 1) & (target == 1)).sum(dim=0).cpu().numpy()
        cls_predicted = (pred == 1).sum(dim=0).cpu().numpy()
        cls_ground = (target == 1).sum(dim=0).cpu().numpy()
        im_correct = ((pred == 1) & (target == 1)).sum(dim=1).cpu().numpy()
        im_predicted = (pred == 1).sum(dim=1).cpu().numpy()
        im_ground = (target == 1).sum(dim=1).cpu().numpy()
        self._cls_correct = np.add(self._cls_correct, cls_correct)
        self._cls_pred = np.add(self._cls_pred, cls_predicted)
        self._cls_ground = np.add(self._cls_ground, cls_ground)
        self._im_correct = np.append(self._im_correct, im_correct)
        self._im_pred = np.append(self._im_pred, im_predicted)
        self._im_ground = np.append(self._im_ground, im_ground)
    
    def reset(self):
        self._cls_correct = np.zeros(self.num_classes)
        self._cls_pred = np.zeros(self.num_classes)
        self._cls_ground = np.zeros(self.num_classes)
        self._im_correct = np.array([])
        self._im_pred = np.array([])
        self._im_ground = np.array([])
    
    def CP(self):
        val = np.divide(
            self.cls_correct, 
            self.cls_pred,
            out = np.zeros_like(self.cls_correct),
            where = (self.cls_pred != 0),
        )
        return val.mean()
    
    def CR(self):
        val = np.divide(
            self.cls_correct, 
            self.cls_ground,
            out = np.zeros_like(self.cls_correct),
            where = (self.cls_ground != 0),
        )
        return val.mean()
    
    def CF1(self):
        cp = self.CP()
        cr = self.CR()
        return 2 * (cp * cr) / (cp + cr + 1e-12)
    
    def IP(self):
        return self.im_correct.sum() / (self.im_pred.sum() + 1e-12)
    
    def IR(self):
        return self.im_correct.sum() / (self.im_ground.sum() + 1e-12)
    
    def IF1(self):
        ip = self.IP()
        ir = self.IR()
        return 2 * (ip * ir) / (ip + ir + 1e-12)
    
    def HF1(self):
        cf1 = self.CF1()
        if1 = self.IF1()
        return 2 * (cf1 * if1) / (cf1 + if1 + 1e-12)


class TagTransform:
    
    def __init__(self, file: str | Path):
        self._tags = pl.read_ndjson(file)
        tags = set(self._tags["name"].to_list())
        tokenizer = get_tokenizer("basic_english")
        self.voc = build_vocab_from_iterator(
            [tokenizer(tag) for tag in tags],
        )
        self.tags = set(self.voc.get_stoi().keys())
        self.voclen = len(self.voc)
        
    def __call__(self, sample: Iterable) -> Tensor:
        return self.encode(sample)
    
    def encode(self, sample: Iterable) -> Tensor:
        ftags = [t for t in sample if t in self.tags]
        if ftags == []:
            return torch.zeros(self.voclen)
        val = F.one_hot(
            torch.tensor(self.voc.forward(ftags)), 
            num_classes=self.voclen
        ).amax(dim=0)
        return val
    
    def decode(self, cls: Tensor) -> list[list[str]]:
        bins = torch.argwhere(cls == 1)
        d = defaultdict(list)
        for k, v in bins:
            d[k.item()].append(v.item())
        ltokens = list(d.values())
        out = []
        for tokens in ltokens:
            out.append(self.voc.lookup_tokens(tokens))
        return out

    def decode_topn(self, cls: Tensor, clen: Tensor) -> Tensor:
        sort_cls = cls.argsort(dim=1, descending=True)
        for i, (_class, _len) in enumerate(zip(sort_cls, clen)):
            iones = _class[:_len]
            cls[i] = torch.zeros_like(cls[i])
            cls[i] = cls[i].put(iones, torch.ones(len(iones), device=cls.device, dtype=cls.dtype))
        return cls
