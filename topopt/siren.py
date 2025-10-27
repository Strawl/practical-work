import jax
import jax.numpy as jnp
import math
import equinox as eqx
from typing import List


def get_siren_weights_init_fun(omega: float, first_layer: bool = False):
    def init_fun(key: jax.random.PRNGKey, shape: tuple, dtype=jnp.float32):
        fan_in, _ = shape[-2:]
        limit = 1. / fan_in if first_layer else math.sqrt(6. / fan_in) / omega
        return jax.random.uniform(key, shape, dtype, minval=-limit, maxval=limit)
    return init_fun


def siren_bias_init(key: jax.random.PRNGKey, shape: tuple, dtype=jnp.float32):
    fan_in = fan_out = shape[-1]
    limit = math.sqrt(1. / fan_in)
    return jax.random.uniform(key, (fan_out,), dtype, minval=-limit, maxval=limit)


class SIREN(eqx.Module):
    weights: List[jax.Array]
    biases: List[jax.Array]
    omega: float
    plain_last: bool

    def __init__(
        self,
        num_channels_in: int,
        num_channels_out: int,
        num_layers: int,
        num_latent_channels: int,
        omega: float,
        rng_key: jax.random.PRNGKey,
        plain_last: bool = True
    ):
        self.omega = omega
        self.plain_last = plain_last

        # define layer sizes
        channels = (num_channels_in, *[num_latent_channels] * (num_layers - 1), num_channels_out)

        # initialize weights and biases for each layer
        keys = jax.random.split(rng_key, 2 * (len(channels) - 1))
        weight_keys = keys[: len(channels) - 1]
        bias_keys = keys[len(channels) - 1 :]

        weights, biases = [], []
        is_first = True
        for (in_c, out_c, wk, bk) in zip(channels[:-1], channels[1:], weight_keys, bias_keys):
            w_init = get_siren_weights_init_fun(omega, first_layer=is_first)
            weights.append(w_init(wk, (in_c, out_c)))
            biases.append(siren_bias_init(bk, (out_c,)))
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
        if not self.plain_last:
            x = self._activation(x)
        return x