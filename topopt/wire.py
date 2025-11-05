import math
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx


# ---------- Simple real Linear: x @ W + b (batch-first) ----------

class Linear(eqx.Module):
    weight: jnp.ndarray  # (in_features, out_features)
    bias: jnp.ndarray | None
    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)

    def __init__(self, in_features, out_features, key):
        self.in_features = in_features
        self.out_features = out_features

        k_w, k_b = jax.random.split(key)
        fan_in = in_features
        scale = 1.0 / math.sqrt(fan_in)

        self.weight = jax.random.normal(k_w, (in_features, out_features)) * scale
        self.bias = jax.random.normal(k_b, (out_features,)) * scale

    def __call__(self, x, *, key=None):
        y = x @ self.weight + self.bias
        return y


# ---------- Complex Linear: x @ W + b, batch-first ----------

class ComplexLinear(eqx.Module):
    weight: jnp.ndarray  # complex, (in_features, out_features)
    bias: jnp.ndarray | None
    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)

    def __init__(self, in_features, out_features, *, key):
        self.in_features = in_features
        self.out_features = out_features

        k1, k2, k3, k4 = jax.random.split(key, 4)

        fan_in = in_features
        scale = 1.0 / math.sqrt(fan_in)

        w_real = jax.random.normal(k1, (in_features, out_features)) * scale
        w_imag = jax.random.normal(k2, (in_features, out_features)) * scale
        self.weight = w_real + 1j * w_imag

        b_real = jax.random.normal(k3, (out_features,)) * scale
        b_imag = jax.random.normal(k4, (out_features,)) * scale
        self.bias = b_real + 1j * b_imag

    def __call__(self, x, *, key=None):
        y = x @ self.weight + self.bias
        return y


class RealGaborLayer(eqx.Module):
    freqs: Linear
    scale: Linear
    omega_0: float
    scale_0: float

    def __init__(
        self,
        in_features,
        out_features,
        *,
        omega0: float = 10.0,
        sigma0: float = 10.0,
        bias: bool = True,
        key,
    ):
        k1, k2 = jax.random.split(key)
        self.freqs = Linear(in_features, out_features, use_bias=bias, key=k1)
        self.scale = Linear(in_features, out_features, use_bias=bias, key=k2)
        self.omega_0 = omega0
        self.scale_0 = sigma0

    def __call__(self, x, *, key=None):
        omega = self.omega_0 * self.freqs(x)
        scale = self.scale(x) * self.scale_0
        return jnp.cos(omega) * jnp.exp(-(scale ** 2))

class ComplexGaborLayer(eqx.Module):
    linear: eqx.Module
    omega_0: jnp.ndarray
    scale_0: jnp.ndarray
    is_first: bool = eqx.field(static=True)

    def __init__(
        self,
        in_features,
        out_features,
        key: jax.random.PRNGKey,
        omega0: float = 10.0,
        sigma0: float = 40.0,
        is_first: bool = False,
    ):
        self.is_first = is_first

        self.omega_0 = jnp.array(omega0, dtype=jnp.float32)
        self.scale_0 = jnp.array(sigma0, dtype=jnp.float32)

        if is_first:
            self.linear = Linear(in_features, out_features, key=key)
        else:
            self.linear = ComplexLinear(in_features, out_features, key=key)

    def __call__(self, x, *, key=None):
        lin = self.linear(x)  # (..., out_features)

        if self.is_first:
            lin = jnp.asarray(lin, dtype=jnp.complex64)

        omega = self.omega_0 * lin
        scale = self.scale_0 * lin

        return jnp.exp(1j * omega - jnp.abs(scale) ** 2)


# ---------- INR / WIRE Network ----------

class WIRE(eqx.Module):
    net: eqx.nn.Sequential
    wavelet: str = eqx.field(static=True, default="gabor")
    complex: bool = eqx.field(static=True, default=True)

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        hidden_layers: int,
        out_features: int,
        rng_key: jax.random.PRNGKey,
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
        scale: float = 10.0,
    ):
        # match your earlier choice: reduce width for complex numbers
        hidden_features = int(hidden_features / np.sqrt(2))

        keys = jax.random.split(rng_key, hidden_layers + 2)
        layers = []

        layers.append(
            ComplexGaborLayer(
                in_features=in_features,
                out_features=hidden_features,
                omega0=first_omega_0,
                sigma0=scale,
                is_first=True,
                key=keys[0],
            )
        )

        for i in range(hidden_layers):
            layers.append(
                ComplexGaborLayer(
                    in_features=hidden_features,
                    out_features=hidden_features,
                    omega0=hidden_omega_0,
                    sigma0=scale,
                    is_first=False,
                    key=keys[i + 1],
                )
            )

        # Final complex linear layer
        final_linear = ComplexLinear(
            hidden_features, out_features, key=keys[-1]
        )
        layers.append(final_linear)

        self.net = eqx.nn.Sequential(layers)

    def __call__(self, coords, *, key=None):
        output = self.net(coords, key=key)
        if self.wavelet == "gabor":
            return jnp.real(output)
        return output