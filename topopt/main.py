"""
Parallel training of multiple SIREN models using vmap.
Each SIREN learns for a different target density (volume fraction).
"""

import config as config
import equinox as eqx
import jax.numpy as np
import optax
from bc import make_bc_preset
from feax.experimental.topopt_toolkit import create_compliance_fn
from feax.mesh import rectangle_mesh
from problems import DensityElasticityProblem
from serialization import (
    ModelEnsembleConfig,
    create_models,
    serialize_ensemble,
)
from tqdm import tqdm
from utils import get_element_centroids

import jax
from feax import (
    DirichletBCConfig,
    DirichletBCSpec,
    InternalVars,
    SolverOptions,
    create_solver,
    zero_like_initial_guess,
)

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

bc_config = DirichletBCConfig(
    [DirichletBCSpec(location=fixed_location, component="all", value=0.0)]
)

problem = DensityElasticityProblem(
    mesh=mesh,
    vec=2,
    dim=2,
    ele_type=ele_type,
    gauss_order=2,
    location_fns=[load_location],
    # additional_info=(70e3, 1e1, 0.3, 3, 1e2)
    additional_info=(70e3, 7, 0.3, 3, 1e2),
)

bc = bc_config.create_bc(problem)
solver_opts = SolverOptions(
    tol=1e-8, linear_solver="bicgstab", use_jacobi_preconditioner=True
)
solver = create_solver(
    problem,
    bc=bc,
    solver_options=solver_opts,
    adjoint_solver_options=solver_opts,
    iter_num=1,
)
initial_guess = zero_like_initial_guess(problem, bc)
traction_array = InternalVars.create_uniform_surface_var(problem, problem.T)
compute_compliance = create_compliance_fn(problem, surface_load_params=problem.T)
num_elements = mesh.cells.shape[0]


def J_total(rho):
    internal_vars = InternalVars(volume_vars=(rho,), surface_vars=[(traction_array,)])
    sol = solver(internal_vars, initial_guess)
    return compute_compliance(sol)


coords = get_element_centroids(mesh)

# ---------------- MODEL Definition ----------------

ensemble_config: ModelEnsembleConfig = ModelEnsembleConfig.from_json(
    config.TRAIN_CONFIG_PATH
)
rng = jax.random.PRNGKey(42)
model_batch, target_densities, penalties = create_models(
    ensemble_config,
    rng,
)

# ---------------- Loss Functions ----------------
def fem_loss_single(model, coords, target_density, lam, penalty):
    rho = jax.nn.sigmoid(model(coords))
    J = J_total(rho)
    mean_rho = np.mean(rho)
    C_raw = mean_rho - target_density
    C = np.maximum(C_raw, 0.0) 
    penalty_term = penalty * C**2
    lagrangian_term = lam * C
    vol_cond = lagrangian_term + penalty_term
    loss = J + vol_cond
    jax.debug.print(
        "mean(rho)={mean_rho:.4f} | C={C:.4f} | J={J:.4f} | vol={vol_cond:.4f} | loss={loss:.4f}",
        mean_rho=mean_rho, C=C, J=J, vol_cond=vol_cond, loss=loss
    )

    return loss, C


single_loss_and_grad = eqx.filter_value_and_grad(fem_loss_single, has_aux=True)


def batched_loss_and_grad(models, coords, target_densities, lams, penalties):
    # vmapped over models/targets/lams/penalties
    (losses, Cs), grads = jax.vmap(
        single_loss_and_grad,
        in_axes=(0, None, 0, 0, 0),
    )(models, coords, target_densities, lams, penalties)
    return losses, Cs, grads


batched_loss_and_grad = jax.jit(batched_loss_and_grad)


# ---------------- Optimizer & Training ----------------
def _optimisation_step(
    models, opt_states, optimizer, coords, target_densities, lams, penalties
):
    losses, Cs, grads = batched_loss_and_grad(
        models, coords, target_densities, lams, penalties
    )
    updates, opt_states = jax.vmap(optimizer.update)(grads, opt_states)
    models = jax.vmap(eqx.apply_updates)(models, updates)
    return models, opt_states, losses, Cs


def optimisation_step(
    models, opt_states, optimizer, coords, target_densities, lams, penalties
):
    losses, Cs, grads = batched_loss_and_grad(
        models, coords, target_densities, lams, penalties
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
    coords,
    target_densities,
    penalties,
    num_epochs=100,
    lr=1e-4,
    patience=5,
    min_delta=0.005,
):
    optimizer = optax.chain(optax.clip_by_global_norm(1.0),
                            optax.adabelief(lr))
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
            models, opt_states, optimizer, coords, target_densities, lams, penalties
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


# ---------------- Train All Models ----------------
trained_models, opt_states = train_multiple_models(
    model_batch, coords, target_densities, penalties, num_epochs=500, lr=1e-4
)

serialize_ensemble(trained_models, opt_states, ensemble_config, problem)
