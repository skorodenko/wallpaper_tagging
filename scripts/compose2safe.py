from pathlib import Path
from models.lp import LP
from models.mlp import MLP
from models.lqp import LQP
from models.vcnn import VCNN
from models.compose import Model
from safetensors.torch import save_file


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

ignore = ["lp.mlp", "lp.vcnn", "lqp.mlp", "lqp.vcnn"]
state = model.state_dict()

to_del = []
for k in state.keys():
    for ik in ignore:
        if ik in k:
            to_del.append(k)
            
for k in to_del:
    state.pop(k)

save_file(state, MODEL_ROOT / "compose.test" / "compose.safetensors")
