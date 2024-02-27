import lightning as lg
from data import DataModule
from pathlib import Path
from models.mlp import MLP
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


TRAINED_MODELS = Path("./assets/trained_models")
ROOT_DIR = TRAINED_MODELS / "mlp.train"


data = DataModule(load_images = False, batch_size = 16, prefetch_factor = 4, num_workers = 7)
trainer = lg.Trainer(
    devices=1,
    max_epochs=100,
    accelerator="gpu",
    logger=CSVLogger(ROOT_DIR, "logs"),
    callbacks=[
        ModelSummary(max_depth=2),
        ModelCheckpoint(
            monitor="val_loss",
            save_top_k=4,
            dirpath= ROOT_DIR / "checkpoints",
            filename="{epoch}---{val_loss:.4f}",
        ),
        EarlyStopping(
            patience=5,
            monitor="val_loss", 
            mode="min",
            min_delta=0.01,
        ),
    ],
)
model = MLP(
    lr = 0.0001,
    weight_decay=1e-4,
    momentum = 0.9,
)
trainer.fit(model, datamodule=data) 
