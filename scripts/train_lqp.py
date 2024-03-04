import lightning as lg
from data import DataModule
from pathlib import Path
from models.lqp import LQP
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


TRAINED_MODELS = Path("./assets/trained_models")
ROOT_DIR = TRAINED_MODELS / "lqp.train"


data = DataModule(batch_size = 16, prefetch_factor = 8, num_workers = 6)
trainer = lg.Trainer(
    devices=1,
    max_epochs=100,
    accelerator="gpu",
    default_root_dir = ROOT_DIR,
    limit_train_batches=0.1,
    logger=CSVLogger(ROOT_DIR, "logs"),
    callbacks=[
        ModelSummary(max_depth=2),
        ModelCheckpoint(
            monitor="val_loss",
            save_top_k=2,
            dirpath=ROOT_DIR / "checkpoints",
            filename="{epoch}---{val_loss:.2f}",
        ),
        EarlyStopping(
            monitor="val_loss", 
            mode="min"
        ),
    ],
)
model = LQP(
    lr = 0.0005,
    weight_decay=0.001,
)
trainer.fit(model, datamodule=data) 
