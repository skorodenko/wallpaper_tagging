import torch
import lightning as lg
from pathlib import Path
from src.data import DataModule
from src.tagattention import TagAttention
from safetensors.torch import load_model
from lightning.pytorch.loggers import CSVLogger


ASSETS = Path("./assets/testout")
OUT_DIR = ASSETS / "compose.test"


data = DataModule(batch_size=32, num_workers=6)
trainer = lg.Trainer(
    devices=1,
    accelerator="gpu",
    limit_test_batches=0.1,
    default_root_dir = OUT_DIR,
    logger=CSVLogger(OUT_DIR, "logs", version=0),
)


def compose_model(**kwargs):
    model = TagAttention(**kwargs)
    load_model(model, "./compose.safetensors")
    model.freeze()
    return model


if __name__ == "__main__":
    model = compose_model()
    with torch.no_grad():
        trainer.test(model, datamodule=data) 
