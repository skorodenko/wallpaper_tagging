import lightning as lg
from pathlib import Path
from data import DataModule
from models.vcnn import VCNN
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelSummary


TRAINED_MODELS = Path("./assets/trained_models")
ROOT_DIR = TRAINED_MODELS / "vcnn.test"
CKPT_PATH = TRAINED_MODELS / "vcnn.test" / "vcnn.ckpt"


data = DataModule(batch_size = 32, prefetch_factor = 16, num_workers = 6)

trainer = lg.Trainer(
    devices = 1,
    max_epochs = 32,
    accelerator = "gpu",
    default_root_dir = ROOT_DIR,
    logger = CSVLogger(ROOT_DIR, "logs", version=0),
    callbacks = [
        ModelSummary(2),
    ],
)

model = VCNN.load_from_checkpoint(CKPT_PATH)

trainer.test(model, datamodule=data)
