import lightning as lg
from data import DataModule
from pathlib import Path
from models.mlp import MLP
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


lg.seed_everything(42)

TRAINED_MODELS = Path("./assets/trained_models")


data = DataModule(batch_size = 32)
trainer = lg.Trainer(
    devices=1,
    max_epochs=500,
    accelerator="auto",
    default_root_dir = TRAINED_MODELS / "mlp.ckpt",
    precision="16-mixed",
    callbacks=[
        EarlyStopping(monitor="val_loss", mode="min"),
    ],
)
model = MLP(
    lr = 0.001,
    weight_decay=0.0001,
)
trainer.fit(model, datamodule=data) 
