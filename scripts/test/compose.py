import torch
import lightning as lg
from pathlib import Path
from data import DataModule
from models.compose import Model
from safetensors.torch import load_model
from lightning.pytorch.loggers import CSVLogger


MODEL_ROOT = Path("./assets/trained_models")
ROOT_DIR = MODEL_ROOT / "compose.test"


data = DataModule(batch_size=32, num_workers=6)
trainer = lg.Trainer(
    devices=1,
    accelerator="gpu",
    default_root_dir = ROOT_DIR,
    logger=CSVLogger(ROOT_DIR, "logs", version=0),
)

model = Model()
load_model(model, ROOT_DIR / "compose.safetensors", strict=False)
model.eval()

if __name__ == "__main__":
    with torch.no_grad():
        trainer.test(model, datamodule=data) 
