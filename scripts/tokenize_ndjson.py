import pathlib
import torch
import polars as pl
import torch.nn.functional as F
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


DATA_ROOT = pathlib.Path("./assets/preprocessed")
TRAIN_OUTPUT = DATA_ROOT / "Train_encoded.ndjson"
TEST_OUTPUT = DATA_ROOT / "Test_encoded.ndjson"
CODE_OUTPUT = DATA_ROOT / "Codes.csv"
TRAIN_FILES = [
    DATA_ROOT / "Train_coco.ndjson",
    DATA_ROOT / "Train_nus-wide.ndjson",
]
TEST_FILES = [
    DATA_ROOT / "Test_coco.ndjson",
    DATA_ROOT / "Test_nus-wide.ndjson"
]

train = pl.scan_ndjson(TRAIN_FILES)
test = pl.scan_ndjson(TEST_FILES)
tags_1k = pl.read_csv("./assets/nus-wide/TagList1k.txt", has_header=False)["column_1"].to_list()
tags_81 = pl.read_csv("./assets/nus-wide/TagList81.txt", has_header=False)["column_1"].to_list()

tags = set(tags_81)
tags.update(tags_1k)
tags = list(tags)

tokenizer = get_tokenizer("basic_english")
voc = build_vocab_from_iterator(
    [tokenizer(tag) for tag in tags],
    specials = [""],
    special_first = True,
)
voc.set_default_index(0)

voc_table = pl.DataFrame(
    enumerate(voc.get_itos()),
    schema = ["code", "token"],
)

def onehot_token(tokens: pl.Series):
    tokens = tokens.to_list()
    val = F.one_hot(
        torch.tensor(voc.forward(tokens)), 
        num_classes=len(voc)
    ).amax(dim=0)
    return pl.Series(val.tolist())
    
train = (
    train 
        .with_columns(
            pl.col("labels").map_elements(
                onehot_token
            )
        )
        .with_columns(
            pl.col("tags").map_elements(
                onehot_token
            )
        )
)

test = (
    test 
        .with_columns(
            pl.col("labels").map_elements(
                onehot_token
            )
        )
        .with_columns(
            pl.col("tags").map_elements(
                onehot_token
            )
        )
)
    
train.collect().write_ndjson(TRAIN_OUTPUT)
test.collect().write_ndjson(TEST_OUTPUT)
voc_table.write_csv(CODE_OUTPUT)
