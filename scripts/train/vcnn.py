import lightning as lg
from data import DataModule
from pathlib import Path
from itertools import chain
from models.vcnn import VCNN
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint


TRAINED_MODELS = Path("./assets/trained_models")
ROOT_DIR = TRAINED_MODELS / "vcnn.train"
CKPT_PATH = TRAINED_MODELS / "vcnn.train" / "vcnn.ckpt"


data = DataModule(batch_size = 32, prefetch_factor = 8, num_workers = 6)

trainer = lg.Trainer(
    devices = 1,
    max_epochs = 40,
    accelerator = "gpu",
    default_root_dir = ROOT_DIR,
    logger = CSVLogger(ROOT_DIR, "logs"),
    limit_train_batches = 0.1,
    limit_val_batches = 0.1,
    callbacks = [
        ModelCheckpoint(
            monitor="val_loss",
            save_top_k=4,
            dirpath=ROOT_DIR / "checkpoints",
            filename="{v_num}-{epoch}-{val_loss:.3f}-[{C_F1:.3f}, {I_F1:.3f}]",
        ),
    ],
)

model = VCNN(
    lr = 0.1,
    weight_decay = 0,
)

freeze_layers = [
    model.conv1.parameters(),
    model.layer1.parameters(),
    model.layer2.parameters(),
    model.layer3.parameters(),
    #model.layer4.parameters(),
]

for param in chain(*freeze_layers):
    param.requires_grad = False

trainer.fit(model, datamodule=data)
