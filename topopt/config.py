from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import jax

TRAIN_CONFIG_PATH = os.getenv("TRAIN_CONFIG_PATH", "./train_configs/train_siren.yaml")
basefilename = Path(TRAIN_CONFIG_PATH).stem

timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
default_dir = Path("./outputs") / f"{timestamp}_{basefilename}"

SAVE_DIR = Path(os.environ.get("SAVE_DIR", str(default_dir))).expanduser().resolve()
SAVE_DIR.mkdir(parents=True, exist_ok=True)

print(f"Saving data to: {SAVE_DIR}")


# ---------------- JAX Config ----------------
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")
jax.config.update("jax_debug_nans", False)
jax.config.update("jax_debug_infs", False)
jax.config.update("jax_traceback_filtering", "off")
print(f"JAX is running on the {jax.default_backend()} backend.")
