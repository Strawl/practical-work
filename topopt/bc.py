import jax.numpy as jnp


def make_bc_preset(name, Lx, Ly):
    """
    Return (fixed_location, load_location) callables for a given BC preset.

    Each callable takes a point = (x, y) and returns a boolean.
    """

    x_tol = 1e-5 * max(1.0, Lx)
    y_tol = 1e-5 * max(1.0, Ly)

    if name == "cantilever_corner":
        # left edge fixed, load at bottom-right corner
        def fixed_location(point):
            x, y = point
            return jnp.isclose(x, 0.0, atol=x_tol)

        def load_location(point):
            x, y = point
            return jnp.logical_and(
                jnp.isclose(x, Lx, atol=x_tol),
                jnp.isclose(y, 0.0, atol=0.1 * Ly + y_tol),
            )

    elif name == "cantilever_mid":
        # Left edge fixed, load at mid-height on the right edge
        def fixed_location(point):
            x, y = point
            return jnp.isclose(x, 0.0, atol=x_tol)

        def load_location(point):
            x, y = point
            return jnp.logical_and(
                jnp.isclose(x, Lx, atol=x_tol),
                jnp.abs(y - 0.5 * Ly) <= 0.05 * Ly + y_tol,
            )

    else:
        raise ValueError(f"Unknown BC preset: {name}")

    return fixed_location, load_location
