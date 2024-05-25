import lightning as lg
from pathlib import Path
from argparse import Namespace
from src.data import DataModule
from src.tagattention import TagAttention
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, ModelSummary


TRAINOUT = Path("./assets/trainout")
CKPT_PATH = TRAINOUT / "tagattention.ckpt"


data = DataModule(batch_size = 32, prefetch_factor = 8, num_workers = 6)


trainer = lg.Trainer(
    devices = 1,
    max_epochs = 30,
    accelerator = "gpu",
    precision = "bf16-mixed",
    default_root_dir = TRAINOUT,
    logger = CSVLogger(TRAINOUT, "logs", version=0),
    limit_train_batches = 0.2,
    limit_val_batches = 0.2,
    callbacks = [
        ModelSummary(2),
        LearningRateMonitor(logging_interval = "step"),
        ModelCheckpoint(
            monitor="H_F1",
            mode="max",
            save_weights_only=True,
            save_top_k=3,
            dirpath=TRAINOUT / "checkpoints",
            save_on_train_epoch_end=True,
            filename="{H_F1:.5f}@{v_num}@{epoch}@{val_loss:.3f}",
        ),
    ],
)

params = Namespace(
    lr = 0.001,
    weight_decay = 0.1,
    sched_gamma = 0.25,
)

model = TagAttention(
    params = params
)

trainer.fit(model, datamodule=data)
