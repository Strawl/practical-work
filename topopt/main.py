"""
Parallel training of multiple SIREN models using vmap.
Each SIREN learns for a different target density (volume fraction).
"""

import sys
from typing import Callable, Optional
import config as config
import equinox as eqx
import jax.numpy as np
import optax
from bc import make_bc_preset
from feax.mesh import rectangle_mesh, Mesh
from fem_utils import create_J_total, get_element_areas, get_element_geometry 
from serialization import (
    ModelEnsembleConfig,
    create_models,
    serialize_ensemble,
)
from tqdm import tqdm

import jax

# ---------------- Loss Functions ----------------
def loss(
    model, target_density, lam, penalty, coords, areas, total_area, complience_func
):
    rho = jax.nn.sigmoid(model(coords))
    J = complience_func(rho)
    mean_rho = np.sum(rho.squeeze() * areas) / total_area
    C_raw = mean_rho - target_density
    C = np.maximum(C_raw, 0.0)
    penalty_term = penalty * C**2
    lagrangian_term = lam * C
    vol_cond = lagrangian_term + penalty_term
    loss = J + vol_cond
    jax.debug.print(
        "mean(rho)={mean_rho:.4f} | C={C:.4f} | J={J:.4f} | vol={vol_cond:.4f} | loss={loss:.4f}",
        mean_rho=mean_rho,
        C=C,
        J=J,
        vol_cond=vol_cond,
        loss=loss,
    )

    return loss, C


loss_and_grad = eqx.filter_value_and_grad(loss, has_aux=True)


def batched_loss_and_grad(
    models,
    target_densities,
    lams,
    penalties,
    coords,
    areas,
    total_area,
    complience_func,
):
    (losses, Cs), grads = jax.vmap(
        loss_and_grad,
        in_axes=(0, 0, 0, 0, None, None, None, None),
    )(
        models,
        target_densities,
        lams,
        penalties,
        coords,
        areas,
        total_area,
        complience_func,
    )
    return losses, Cs, grads


batched_loss_and_grad = jax.jit(
    batched_loss_and_grad,
    static_argnames=(
        "total_area",
        "complience_func",
    ),
)


def optimisation_step(
    models,
    opt_states,
    optimizer,
    target_densities,
    lams,
    penalties,
    coords,
    areas,
    total_area,
    complience_func,
):
    losses, Cs, grads = batched_loss_and_grad(
        models,
        target_densities,
        lams,
        penalties,
        coords,
        areas,
        total_area,
        complience_func,
    )

    updates, new_opt_states = jax.vmap(optimizer.update)(grads, opt_states, value=losses)

    new_models = jax.vmap(eqx.apply_updates)(models, updates)

    return new_models, new_opt_states, losses, Cs


def train_multiple_models(
    models,
    mesh: Mesh,
    target_densities: np.ndarray,
    penalties: np.ndarray,
    complience_func: Callable[[np.ndarray], float],
    num_epochs: int,
    lr: float,
    jitted_coords: bool,
):
    # For monitoring
    lr_scale_history = []
    loss_history = []
    lam_history = []
    Cs_history = []

    # Mesh
    geom = get_element_geometry(mesh)
    coords = centroids_scaled = geom["centroids_scaled"]
    num_elements = geom["num_elements"]
    dx = geom["dx_scaled"]
    dy = geom["dy_scaled"]
    areas, total_area = get_element_areas(mesh)

    PATIENCE = 20
    COOLDOWN = 0
    FACTOR = 0.5
    RTOL = 5e-2
    ACCUMULATION_SIZE = 1

    # Optimizer
    inner_optim = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adabelief(lr),
        optax.contrib.reduce_on_plateau(
            patience=PATIENCE,
            cooldown=COOLDOWN,
            factor=FACTOR,
            rtol=RTOL,
            accumulation_size=ACCUMULATION_SIZE,
        ),
    )

    optimizer = optax.apply_if_finite(
        inner_optim,
        max_consecutive_errors=30,
    )
    opt_states = jax.vmap(lambda m: optimizer.init(eqx.filter(m, eqx.is_array)))(models)


    rng = jax.random.PRNGKey(42)
    lams = np.zeros_like(target_densities)

    for epoch in tqdm(range(num_epochs), desc="Epochs"):

        if jitted_coords:
            # uniform jitter in [-0.5, 0.5] * element size
            rng, kx, ky = jax.random.split(rng, 3)
            ux = jax.random.uniform(kx, (num_elements,), minval=-0.5, maxval=0.5)
            uy = jax.random.uniform(ky, (num_elements,), minval=-0.5, maxval=0.5)

            coords = np.stack(
                [
                    centroids_scaled[:, 0] + ux * dx,
                    centroids_scaled[:, 1] + uy * dy,
                ],
                axis=-1,
            )

        models, opt_states, losses, Cs = optimisation_step(
            models,
            opt_states,
            optimizer,
            target_densities,
            lams,
            penalties,
            coords,
            areas,
            int(total_area),
            complience_func=complience_func,
        )

        good = np.isfinite(Cs)
        lams = np.where(good, lams + 2.0 * penalties * Cs, lams)

        # Monitoring for inspecting later on
        
        Cs_history.append(Cs)
        lam_history.append(lams)
        loss_values = np.array(losses)
        loss_history.append(loss_values)
        lr_scales = jax.vmap(lambda s: optax.tree.get(s, "scale"))(opt_states)
        lr_scale_history.append(lr_scales)

        # Printing while training
        loss_str = " | ".join(
            [f"{loss_values[i]:.6f}" for i in range(len(loss_values))]
        )
        mean_loss = float(np.mean(loss_values))
        print(
            f"Epoch {epoch:03d} | mean loss = {mean_loss:.6f} | individual = [{loss_str}]"
        )


    history = (np.array(loss_history), np.array(lr_scale_history), np.array(lam_history), np.array(Cs_history))
    return models, opt_states, history


def main(train_config_path: Optional[str] = config.TRAIN_CONFIG_PATH):
    # ---------------- FE Problem Setup ----------------
    ele_type = "QUAD4"
    Lx, Ly = 60.0, 30.0
    scale = 1
    Nx = 60 * scale
    Ny = 30 * scale
    mesh = rectangle_mesh(Nx, Ny, domain_x=Lx, domain_y=Ly)

    fixed_location, load_location = make_bc_preset("cantilever_corner", Lx, Ly)

    J_total = create_J_total(mesh, fixed_location, load_location, ele_type=ele_type)

    # ---------------- MODEL Definition ----------------

    ensemble_config: ModelEnsembleConfig = ModelEnsembleConfig.from_json(
        train_config_path
    )
    rng = jax.random.PRNGKey(11)
    model_batch, target_densities, penalties = create_models(
        ensemble_config,
        rng,
    )

    trained_models, opt_states, history = train_multiple_models(
        models=model_batch,
        mesh=mesh,
        target_densities=target_densities,
        penalties=penalties,
        complience_func=J_total,
        num_epochs=100,
        lr=1e-3,
        jitted_coords=False,
    )
    serialize_ensemble(trained_models, opt_states, ensemble_config, history)


if __name__ == "__main__":
    cli_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(cli_path)
