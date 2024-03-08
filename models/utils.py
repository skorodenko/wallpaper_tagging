import torch
import numpy as np
import polars as pl
from torch import Tensor
from pandas import Series
import torch.nn.functional as F
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


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
        if cp + cr == 0:
            return 0
        return 2 * (cp * cr) / (cp + cr)
    
    def IP(self):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.nan_to_num(self.im_correct.sum() / self.im_pred.sum())
    
    def IR(self):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.nan_to_num(self.im_correct.sum() / self.im_ground.sum())
    
    def IF1(self):
        ip = self.IP()
        ir = self.IR()
        if ip + ir == 0:
            return 0
        return 2 * (ip * ir) / (ip + ir)


class TagEncoder:
    
    def __init__(self):
        FILE = "./assets/preprocessed/Tags1k.ndjson"
        tags = pl.read_ndjson(FILE)
        tags = set(tags["name"].to_list())
        
        tokenizer = get_tokenizer("basic_english")
        self.voc = build_vocab_from_iterator(
            [tokenizer(tag) for tag in tags],
        )
        self.tags = set(self.voc.get_stoi().keys())
        self.voclen = len(self.voc)
        
    def __call__(self, sample: Series) -> Tensor:
        ftags_1k = [t for t in sample if t in self.tags]
        if ftags_1k == []:
            return torch.zeros(self.voclen)
        val = F.one_hot(
            torch.tensor(self.voc.forward(ftags_1k)), 
            num_classes=self.voclen
        ).amax(dim=0)
        return val
    
    def decode(self, cls: Tensor, clen: Tensor) -> list[str]:
        sort_cls = cls.argsort(dim=1, descending=True)
        out = []
        for _class, _len in zip(sort_cls, clen):
            print(self.voc.lookup_tokens(_class[:10].tolist()))
            v = _class[:_len]
            d = self.voc.lookup_tokens(v.tolist())
            out.append(d)
        
        return out
