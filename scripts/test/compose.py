import torch
import lightning as lg
from pathlib import Path
from data import DataModule
from models.compose import Model
from safetensors.torch import load_model


MODEL_ROOT = Path("./assets/trained_models")
ROOT_DIR = MODEL_ROOT / "compose.test"


data = DataModule(batch_size=1, num_workers=0)
trainer = lg.Trainer(
    devices=1,
    limit_test_batches=4,
    accelerator="gpu",
    default_root_dir = ROOT_DIR,
)

model = Model()
load_model(model, ROOT_DIR / "compose.safetensors")
with torch.no_grad():
    trainer.test(model, datamodule=data) 
