import lightning as lg
from pathlib import Path
from data import DataModule
from models.vcnn import VCNN
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, ModelSummary


TRAINED_MODELS = Path("./assets/trained_models")
ROOT_DIR = TRAINED_MODELS / "vcnn.train"
CKPT_PATH = TRAINED_MODELS / "vcnn.train" / "vcnn.ckpt"


data = DataModule(batch_size = 32, prefetch_factor = 8, num_workers = 6)

trainer = lg.Trainer(
    devices = 1,
    max_epochs = 10,
    accelerator = "gpu",
    default_root_dir = ROOT_DIR,
    logger = CSVLogger(ROOT_DIR, "logs", version=0),
    limit_train_batches = 1,
    limit_val_batches = 1,
    callbacks = [
        ModelSummary(2),
        LearningRateMonitor(logging_interval = "step"),
        ModelCheckpoint(
            monitor="val_loss",
            save_top_k=4,
            dirpath=ROOT_DIR / "checkpoints",
            filename="{v_num}-{epoch}-{val_loss:.3f}-[{H_F1:.3f}]",
        ),
    ],
)

model = VCNN(
    lr = 0.001,
    weight_decay = 0.9997,
)

freeze_layers = ["conv1", "bn1", "layer1", "layer2", "layer3"]

for name, param in model.backbone.named_parameters():
    for freeze in freeze_layers:
        if freeze in name:
            param.requires_grad = False

trainer.fit(model, datamodule=data)
