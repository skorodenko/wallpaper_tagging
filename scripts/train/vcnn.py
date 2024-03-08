import lightning as lg
from data import DataModule
from pathlib import Path
from models.vcnn import VCNN
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary


TRAINED_MODELS = Path("./assets/trained_models")
ROOT_DIR = TRAINED_MODELS / "vcnn.train"
CKPT_PATH = TRAINED_MODELS / "vcnn.train" / "vcnn.ckpt"


data = DataModule(batch_size = 8, prefetch_factor = 8, num_workers = 6)

trainer = lg.Trainer(
    devices = 1,
    max_epochs = 40,
    accelerator = "gpu",
    default_root_dir = ROOT_DIR,
    logger = CSVLogger(ROOT_DIR, "logs"),
    limit_train_batches = 0.05,
    limit_val_batches = 0.2,
    callbacks = [
        ModelSummary(max_depth=1),
        ModelCheckpoint(
            monitor="val_loss",
            save_top_k=4,
            dirpath=ROOT_DIR / "checkpoints",
            filename="{epoch}---{val_loss:.2f}",
        ),
    ],
)

model = VCNN.load_from_checkpoint(
    CKPT_PATH,
    lr = 0.001,
    weight_decay = 0.1,
)

trainer.fit(model, datamodule=data)
