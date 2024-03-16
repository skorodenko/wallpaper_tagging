import torch
import lightning as lg
from pathlib import Path
from data import DataModule
from models.compose import Model


TRAINED_MODELS = Path("./assets/trained_models")
ROOT_DIR = TRAINED_MODELS / "lp.train"


data = DataModule()
trainer = lg.Trainer(
    devices=1,
    accelerator="gpu",
    default_root_dir = ROOT_DIR,
)

with torch.no_grad():
    model = Model()
    trainer.test(model, datamodule=data) 