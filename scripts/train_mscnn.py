import lightning as lg
from itertools import chain
from data import DataModule
from pathlib import Path
from models.mscnn import MSCNN
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


TRAINED_MODELS = Path("./assets/trained_models")
ROOT_DIR = TRAINED_MODELS / "mscnn.train"


data = DataModule(batch_size = 8, prefetch_factor = 4, num_workers = 6)
trainer = lg.Trainer(
    devices=1,
    max_epochs=40,
    accelerator="gpu",
    limit_train_batches=0.05,
    default_root_dir = ROOT_DIR,
    logger=CSVLogger(ROOT_DIR, "logs"),
    callbacks=[
        ModelSummary(max_depth=2),
        ModelCheckpoint(
            monitor="val_loss",
            save_top_k=4,
            dirpath=ROOT_DIR / "checkpoints",
            filename="{epoch}---{val_loss:.2f}",
        ),
        EarlyStopping(
            patience=2,
            min_delta=0.01,
            monitor="val_loss", 
            mode="min",
        ),
    ],
)
model = MSCNN(
    lr = 0.002,
    weight_decay=1e-4,
    momentum = 0.9,
)

# Freeze parameters
freeze_params = [
    model.conv1.parameters(),
    model.fb1.main_layer.parameters(),
    model.fb2.main_layer.parameters(),
    model.fb3.main_layer.parameters(),
    model.fb4.main_layer.parameters(),
]
for param in chain(*freeze_params):
    param.requires_grad = False

trainer.fit(model, datamodule=data)
