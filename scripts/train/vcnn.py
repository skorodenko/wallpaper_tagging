import lightning as lg
from pathlib import Path
from data import DataModule
from models.vcnn import VCNN
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, ModelSummary


TRAINED_MODELS = Path("./assets/trained_models")
ROOT_DIR = TRAINED_MODELS / "vcnn.train"
CKPT_PATH = TRAINED_MODELS / "vcnn.train" / "vcnn.ckpt"


data = DataModule(batch_size = 32, prefetch_factor = 16, num_workers = 6)

trainer = lg.Trainer(
    devices = 1,
    max_epochs = 3,
    accelerator = "gpu",
    default_root_dir = ROOT_DIR,
    logger = CSVLogger(ROOT_DIR, "logs", version=0),
    limit_train_batches = 1.0,
    limit_val_batches = 1.0,
    callbacks = [
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
    ],
)

model = VCNN(
    lr = 0.001,
    weight_decay = 0.0003,
)

freeze_ignore = ["fc"]
freeze_layers = ["base"]

for name, param in model.named_parameters():
    for freeze in freeze_layers:
        if freeze in name and "fc" not in name:
            param.requires_grad = False

trainer.fit(model, datamodule=data)
