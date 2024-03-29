import lightning as lg
from data import DataModule
from pathlib import Path
from models.mlp import MLP
from models.vcnn import VCNN
from models.lqp import LQP
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary, LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


TRAINED_MODELS = Path("./assets/trained_models")
ROOT_DIR = TRAINED_MODELS / "lqp.train"


data = DataModule(batch_size = 32, prefetch_factor = 8, num_workers = 6)

trainer = lg.Trainer(
    devices=1,
    max_epochs=4,
    accelerator="gpu",
    default_root_dir = ROOT_DIR,
    logger=CSVLogger(ROOT_DIR, "logs", version=0),
    callbacks=[
        ModelSummary(2),
        LearningRateMonitor(logging_interval = "step"),
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_weights_only=True,
            save_top_k=3,
            dirpath=ROOT_DIR / "checkpoints",
            save_on_train_epoch_end=True,
            filename="{v_num}@{epoch}@{val_loss:.3f}",
        ),
        EarlyStopping(
            monitor="val_loss", 
            mode="min"
        ),
    ],
)

vcnn = VCNN.load_from_checkpoint("./assets/trained_models/vcnn.train/vcnn.ckpt")
vcnn.freeze()
mlp = MLP.load_from_checkpoint("./assets/trained_models/mlp.train/mlp.ckpt")
mlp.freeze()

model = LQP(
    lr = 0.0001,
    weight_decay=0.0003,
    models={"vcnn": vcnn, "mlp": mlp}
)

trainer.fit(model, datamodule=data) 
