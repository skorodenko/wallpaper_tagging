import lightning as lg
from data import DataModule
from pathlib import Path
from models.vcnn import VCNN
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelSummary


TRAINED_MODELS = Path("./assets/trained_models")
ROOT_DIR = TRAINED_MODELS / "vcnn.test"
CKPT_PATH = TRAINED_MODELS / "vcnn.train" / "vcnn.ckpt"


data = DataModule(batch_size = 16, prefetch_factor = 8, num_workers = 6)
trainer = lg.Trainer(
    devices=1,
    accelerator="gpu",
    default_root_dir = ROOT_DIR,
    logger=CSVLogger(ROOT_DIR, "logs"),
    limit_test_batches=0.1,
    callbacks=[
        ModelSummary(max_depth=1),
    ],
)

model = VCNN.load_from_checkpoint(CKPT_PATH)

trainer.test(model, datamodule=data)
