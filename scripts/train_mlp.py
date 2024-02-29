import lightning as lg
from data import DataModule
from pathlib import Path
from models.mlp import MLP
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


lg.seed_everything(42)

TRAINED_MODELS = Path("./assets/trained_models")


data = DataModule(load_images = False, batch_size = 128, prefetch_factor = 10, num_workers = 7)
trainer = lg.Trainer(
    devices=1,
    max_epochs=40,
    accelerator="gpu",
    default_root_dir = TRAINED_MODELS / "mlp.train",
    callbacks=[
        EarlyStopping(monitor="val_loss", mode="min"),
    ],
)
model = MLP(
    lr = 0.0005,
    weight_decay=0.0001,
)
trainer.fit(model, datamodule=data) 
