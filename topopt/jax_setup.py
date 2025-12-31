def init_jax(
    *,
    enable_x64: bool = True,
    precision: str = "highest",
    debug_nans: bool = False,
    debug_infs: bool = False,
    traceback_filtering: str = "off",
):
    import jax

    jax.config.update("jax_enable_x64", enable_x64)
    jax.config.update("jax_default_matmul_precision", precision)
    jax.config.update("jax_debug_nans", debug_nans)
    jax.config.update("jax_debug_infs", debug_infs)
    jax.config.update("jax_traceback_filtering", traceback_filtering)
    print(f"JAX backend: {jax.default_backend()}")
