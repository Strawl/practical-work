import jax

TRAIN_CONFIG_PATH = "train_siren.json"

# ---------------- JAX Config ----------------
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "high")
jax.config.update("jax_debug_nans", False)
jax.config.update("jax_debug_infs", False)
jax.config.update("jax_traceback_filtering", "off")
print(f"JAX is running on the {jax.default_backend()} backend.")
