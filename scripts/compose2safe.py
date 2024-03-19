from pathlib import Path
from models.lp import LP
from models.mlp import MLP
from models.lqp import LQP
from models.vcnn import VCNN
from models.compose import Model
from safetensors.torch import save_model


MODEL_ROOT = Path("./assets/trained_models")


vcnn = VCNN.load_from_checkpoint(MODEL_ROOT / "vcnn.train" / "vcnn.ckpt")
mlp = MLP.load_from_checkpoint(MODEL_ROOT / "mlp.train" / "mlp.ckpt")
lp = LP.load_from_checkpoint(MODEL_ROOT / "lp.train" / "lp.ckpt")
lqp = LQP.load_from_checkpoint(MODEL_ROOT / "lqp.train" / "lqp.ckpt")


model = Model(
    models = {
        "vcnn": vcnn,
        "mlp": mlp,
        "lp": lp,
        "lqp": lqp,
    }
)
model.freeze()

save_model(model, MODEL_ROOT / "compose.test" / "compose.safetensors")
