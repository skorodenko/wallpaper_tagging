import lightning as lg
from data import DataModule
from pathlib import Path
from models.lp import LP
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary, LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


TRAINED_MODELS = Path("./assets/trained_models")
ROOT_DIR = TRAINED_MODELS / "lp.train"
CKPT_PATH = TRAINED_MODELS / "lp.train" / "lp.ckpt"


data = DataModule(batch_size = 32, prefetch_factor = 8, num_workers = 6)

trainer = lg.Trainer(
    devices=1,
    max_epochs=100,
    accelerator="gpu",
    default_root_dir = ROOT_DIR,
    logger=CSVLogger(ROOT_DIR, "logs", version=0),
    limit_train_batches = 1.0,
    limit_val_batches = 1.0,
    callbacks=[
        ModelSummary(2),
        LearningRateMonitor(logging_interval = "step"),
        ModelCheckpoint(
            monitor="H_F1",
            mode="max",
            save_weights_only=True,
            save_top_k=3,
            dirpath=ROOT_DIR / "checkpoints",
            save_on_train_epoch_end=True,
            filename="{H_F1:.3f}@{v_num}@{epoch}@{val_loss:.3f}",
        ),
        EarlyStopping(
            min_delta=0.01,
            monitor="val_loss", 
            mode="min"
        ),
    ],
)

model = LP(
    lr = 0.01,
    weight_decay=0.01,
)

trainer.fit(model, datamodule=data) 
