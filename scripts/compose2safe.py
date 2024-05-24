from pathlib import Path
from src.tagattention import TagAttention
from safetensors.torch import save_model


MODEL_ROOT = Path("./assets/trainout")

compose = TagAttention.load_from_checkpoint(MODEL_ROOT / "compose.ckpt")

save_model(compose, "./compose.safetensors")
