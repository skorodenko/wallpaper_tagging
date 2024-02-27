import lightning as lg
from data import DataModule
from pathlib import Path
from models.mscnn import MSCNN
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


lg.seed_everything(42)

TRAINED_MODELS = Path("./assets/trained_models")


data = DataModule(batch_size = 6)
trainer = lg.Trainer(
    devices=1,
    max_epochs=40,
    accelerator="gpu",
    default_root_dir = TRAINED_MODELS / "mscnn.ckpt",
    callbacks=[
        EarlyStopping(monitor="val_loss", mode="min"),
    ],
)
model = MSCNN(
    lr = 0.005,
    weight_decay=0.0001,
)
trainer.fit(model, datamodule=data) 
