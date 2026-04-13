import math
from typing import List

import equinox as eqx
import jax.numpy as jnp

import jax


def _get_siren_weights_init_fun(omega: float, first_layer: bool = False):
    def init_fun(key: jax.random.PRNGKey, shape: tuple, dtype=jnp.float32):
        fan_in, _ = shape[-2:]
        limit = 1.0 / fan_in if first_layer else math.sqrt(6.0 / fan_in) / omega
        return jax.random.uniform(key, shape, dtype, minval=-limit, maxval=limit)

    return init_fun


def _siren_bias_init(key: jax.random.PRNGKey, shape: tuple, dtype=jnp.float32):
    fan_in = fan_out = shape[-1]
    limit = math.sqrt(1.0 / fan_in)
    return jax.random.uniform(key, (fan_out,), dtype, minval=-limit, maxval=limit)


class SIREN(eqx.Module):
    weights: List[jax.Array]
    biases: List[jax.Array]
    omega: float

    def __init__(
        self,
        *,
        num_input_units: int | None = None,
        num_output_units: int | None = None,
        num_hidden_layers: int | None = None,
        num_hidden_units: int | None = None,
        omega: float,
        rng_key: jax.random.PRNGKey,
        **legacy_kwargs,
    ):
        self.omega = omega

        # Backward-compatible aliases for older configs and scripts.
        if num_input_units is None:
            num_input_units = legacy_kwargs.pop("num_channels_in", None)
        if num_output_units is None:
            num_output_units = legacy_kwargs.pop("num_channels_out", None)
        if num_hidden_layers is None:
            legacy_num_layers = legacy_kwargs.pop("num_layers", None)
            if legacy_num_layers is not None:
                num_hidden_layers = legacy_num_layers - 1
        if num_hidden_units is None:
            num_hidden_units = legacy_kwargs.pop("num_latent_channels", None)

        if legacy_kwargs:
            unexpected = ", ".join(sorted(legacy_kwargs))
            raise TypeError(f"Unexpected keyword arguments: {unexpected}")

        if (
            num_input_units is None
            or num_output_units is None
            or num_hidden_layers is None
            or num_hidden_units is None
        ):
            raise TypeError(
                "SIREN requires num_input_units, num_output_units, "
                "num_hidden_layers, and num_hidden_units."
            )

        # define layer sizes
        units_per_layer = (
            num_input_units,
            *[num_hidden_units] * num_hidden_layers,
            num_output_units,
        )

        # initialize weights and biases for each layer
        keys = jax.random.split(rng_key, 2 * (len(units_per_layer) - 1))
        weight_keys = keys[: len(units_per_layer) - 1]
        bias_keys = keys[len(units_per_layer) - 1 :]

        weights, biases = [], []
        is_first = True
        for num_inputs, num_outputs, wk, bk in zip(
            units_per_layer[:-1], units_per_layer[1:], weight_keys, bias_keys
        ):
            w_init = _get_siren_weights_init_fun(omega, first_layer=is_first)
            weights.append(w_init(wk, (num_inputs, num_outputs)))
            biases.append(_siren_bias_init(bk, (num_outputs,)))
            is_first = False

        self.weights = weights
        self.biases = biases

    def _linear(self, x: jax.Array, w: jax.Array, b: jax.Array):
        return x @ w + b

    def _activation(self, x: jax.Array):
        return jnp.sin(self.omega * x)

    def __call__(self, x: jax.Array):
        # hidden layers
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            x = self._activation(self._linear(x, w, b))
        # last layer (optional activation)
        x = self._linear(x, self.weights[-1], self.biases[-1])
        return x
