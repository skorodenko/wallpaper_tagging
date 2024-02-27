import torch
import json
import polars as pl
import torch.nn.functional as F
from torch import Tensor
from polars import Series
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


class TagEncoder:
    
    def __init__(self):
        FILE = "./assets/coco/annotations/instances_train2017.json"
        with open(FILE) as f:
            instances_file = json.load(f)
        tags_coco = pl.DataFrame(
            instances_file["categories"],
            schema = {
                "supercategory": pl.String,
                "name": pl.String,
            }
        )
        tags_1k = pl.read_csv("./assets/nus-wide/TagList1k.txt", has_header=False)["column_1"].to_list()
        tags_81 = pl.read_csv("./assets/nus-wide/TagList81.txt", has_header=False)["column_1"].to_list()
        
        tags = set(tags_81)
        tags.update(tags_1k)
        tags.update(tags_coco["name"].unique().to_list())
        tags.update(tags_coco["supercategory"].unique().to_list())
        tags = list(tags)
        tokenizer = get_tokenizer("basic_english")
        self.voc = build_vocab_from_iterator(
            [tokenizer(tag) for tag in tags],
            specials = [""],
            special_first = True,
        )
        self.voc.set_default_index(0)
        self.voclen = len(self.voc)
        del instances_file, tags_coco
        
    def __call__(self, sample: Series) -> Tensor:
        tokens = sample.to_list()
        val = F.one_hot(
            torch.tensor(self.voc.forward(tokens)), 
            num_classes=self.voclen
        ).amax(dim=0)
        return val