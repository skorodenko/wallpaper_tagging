import lightning as lg
from pathlib import Path
from data import DataModule
from models.vcnn import VCNN
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, ModelSummary


TRAINED_MODELS = Path("./assets/trained_models")
ROOT_DIR = TRAINED_MODELS / "vcnn.train"
CKPT_PATH = TRAINED_MODELS / "vcnn.train" / "vcnn.ckpt"


data = DataModule(batch_size = 50, prefetch_factor = 16, num_workers = 6)

trainer = lg.Trainer(
    devices = 1,
    max_epochs = 10,
    accelerator = "gpu",
    default_root_dir = ROOT_DIR,
    logger = CSVLogger(ROOT_DIR, "logs", version=0),
    limit_train_batches = 1.0,
    limit_val_batches = 1.0,
    gradient_clip_val = None,
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
    lr = 0.0001,
    weight_decay = 0.0003,
)

#freeze_layers = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]

for name, param in model.named_parameters():
    #for freeze in freeze_layers:
        #if freeze in name:
    param.requires_grad = True

trainer.fit(model, datamodule=data)
