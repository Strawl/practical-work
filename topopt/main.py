"""
Parallel training of multiple SIREN models using vmap.
Each SIREN learns for a different target density (volume fraction).
"""

import config as config
import equinox as eqx
import jax.numpy as np
import optax
from bc import make_bc_preset
from feax.mesh import rectangle_mesh
from fem_utils import create_J_total, get_element_areas, get_element_centroids
from serialization import (
    ModelEnsembleConfig,
    create_models,
    serialize_ensemble,
)
from tqdm import tqdm

import jax

# ---------------- FE Problem Setup ----------------
ele_type = "QUAD4"
Lx, Ly = 60.0, 30.0
scale = 1
Nx = 60 * scale
Ny = 30 * scale
dx = Lx / Nx
dy = Ly / Ny
dx_norm = 2 * dx / Lx
dy_norm = 2 * dy / Ly
mesh = rectangle_mesh(Nx, Ny, domain_x=Lx, domain_y=Ly)

fixed_location, load_location = make_bc_preset("cantilever_corner", Lx, Ly)

J_total = create_J_total(mesh, fixed_location, load_location, ele_type=ele_type)


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

    good = np.isfinite(losses)
    updates, new_opt_states = jax.vmap(optimizer.update)(grads, opt_states)

    def mask_updates(update, keep):
        return jax.tree.map(lambda u: np.where(keep, u, np.zeros_like(u)), update)

    masked_updates = jax.vmap(mask_updates)(updates, good)

    def apply_or_keep(model, masked_update, keep):
        new_model = eqx.apply_updates(model, masked_update)
        return jax.tree.map(lambda new, old: np.where(keep, new, old), new_model, model)

    models = jax.vmap(apply_or_keep)(models, masked_updates, good)

    def mask_opt_state(new, old, keep):
        return jax.tree.map(lambda n, o: np.where(keep, n, o), new, old)

    opt_states = jax.vmap(mask_opt_state)(new_opt_states, opt_states, good)

    return models, opt_states, losses, Cs


def train_multiple_models(
    models,
    target_densities,
    penalties,
    coords,
    areas,
    total_area,
    complienec_func=J_total,
    num_epochs=100,
    lr=1e-4,
    patience=20,
    min_delta=0.005,
):
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adabelief(lr))
    opt_states = jax.vmap(lambda m: optimizer.init(eqx.filter(m, eqx.is_array)))(models)

    # rng = jax.random.PRNGKey(42)
    lams = np.zeros_like(target_densities)

    # Early stopping state
    best_mean_loss = np.inf
    epochs_no_improve = 0

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        # rng, kx, ky = jax.random.split(rng, 3)
        # ux = jax.random.uniform(kx, (num_elements,), minval=-0.5, maxval=0.5)
        # uy = jax.random.uniform(ky, (num_elements,), minval=-0.5, maxval=0.5)

        # jittered_coords = np.stack(
        # [
        # coords[:, 0] + ux * dx_norm,
        # coords[:, 1] + uy * dy_norm,
        # ],
        # axis=-1,
        # )

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
            complience_func=complienec_func,
        )

        good = np.isfinite(Cs)
        lams = np.where(good, lams + 2.0 * penalties * Cs, lams)

        loss_values = np.array(losses)
        mean_loss = float(np.mean(loss_values))

        loss_str = " | ".join(
            [f"{loss_values[i]:.6f}" for i in range(len(loss_values))]
        )
        print(
            f"Epoch {epoch:03d} | mean loss = {mean_loss:.6f} | individual = [{loss_str}]"
        )

        # ----- Early stopping logic -----
        if best_mean_loss - mean_loss > min_delta:
            # Significant improvement
            best_mean_loss = mean_loss
            epochs_no_improve = 0
        else:
            # Little or no improvement
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(
                    f"Early stopping at epoch {epoch:03d} "
                    f"(no significant improvement for {patience} epochs; "
                    f"best mean loss = {best_mean_loss:.6f})"
                )
                break
        # -------------------------------

    return models, opt_states


def main():
    _, coords = get_element_centroids(mesh)
    areas, total_area = get_element_areas(mesh)

    # ---------------- MODEL Definition ----------------

    ensemble_config: ModelEnsembleConfig = ModelEnsembleConfig.from_json(
        config.TRAIN_CONFIG_PATH
    )
    rng = jax.random.PRNGKey(42)
    model_batch, target_densities, penalties = create_models(
        ensemble_config,
        rng,
    )

    trained_models, opt_states = train_multiple_models(
        model_batch,
        target_densities,
        penalties,
        coords,
        areas,
        total_area,
        num_epochs=150,
        lr=1e-4,
    )
    serialize_ensemble(trained_models, opt_states, ensemble_config)


if __name__ == "__main__":
    main()
