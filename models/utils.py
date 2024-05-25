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
    
    def update(self, pred: Tensor, target: Tensor):
        cls_tp = ((pred == 1) & (target == 1)).sum(dim=0).cpu().numpy()
        cls_tn = ((pred == 0) & (target == 0)).sum(dim=0).cpu().numpy()
        cls_fp = ((pred == 1) & (target == 0)).sum(dim=0).cpu().numpy()
        cls_fn = ((pred == 0) & (target == 1)).sum(dim=0).cpu().numpy()
        im_tp = ((pred == 1) & (target == 1)).sum(dim=1).cpu().numpy()
        im_tn = ((pred == 0) & (target == 0)).sum(dim=1).cpu().numpy()
        im_fp = ((pred == 1) & (target == 0)).sum(dim=1).cpu().numpy()
        im_fn = ((pred == 0) & (target == 1)).sum(dim=1).cpu().numpy()
        self._cls_tp = np.add(self._cls_tp, cls_tp)
        self._cls_tn = np.add(self._cls_tn, cls_tn)
        self._cls_fp = np.add(self._cls_fp, cls_fp)
        self._cls_fn = np.add(self._cls_fn, cls_fn)
        self._im_tp = np.append(self._im_tp, im_tp)
        self._im_tn = np.append(self._im_tn, im_tn)
        self._im_fp = np.append(self._im_fp, im_fp)
        self._im_fn = np.append(self._im_fn, im_fn)
    
    def reset(self):
        self._cls_tp = np.zeros(self.num_classes)
        self._cls_tn = np.zeros(self.num_classes)
        self._cls_fp = np.zeros(self.num_classes)
        self._cls_fn = np.zeros(self.num_classes)
        self._im_tp = np.array([])
        self._im_tn = np.array([])
        self._im_fp = np.array([])
        self._im_fn = np.array([])
    
    def CP(self):
        scls_tp_fp = self._cls_tp + self._cls_fp 
        val = np.divide(
            self._cls_tp, 
            scls_tp_fp,
            out = np.zeros_like(self._cls_tp),
            where = (scls_tp_fp != 0),
        )
        return val.mean()
    
    def CR(self):
        scls_tp_fn = self._cls_tp + self._cls_fn
        val = np.divide(
            self._cls_tp, 
            scls_tp_fn,
            out = np.zeros_like(self._cls_tp),
            where = (scls_tp_fn != 0),
        )
        return val.mean()
    
    def CF1(self):
        cp = self.CP()
        cr = self.CR()
        return 2 * (cp * cr) / (cp + cr + 1e-12)
    
    def IP(self):
        sim_tp_fp = self._im_tp + self._im_fp
        return self._im_tp.sum() / (sim_tp_fp.sum() + 1e-12)
    
    def IR(self):
        sim_tp_fn = self._im_tp + self._im_fn
        return self._im_tp.sum() / (sim_tp_fn.sum() + 1e-12)
    
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
