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
from feax.topopt_toolkit import create_compliance_fn
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
mesh = rectangle_mesh(Nx=60 * scale, Ny=30 * scale, domain_x=Lx, domain_y=Ly)

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
@jax.jit
def fem_loss_single(model, coords, target_density, penalty):
    rho = jax.nn.sigmoid(np.nan_to_num(model(coords), nan=0.0, posinf=1.0, neginf=0.0))
    J = J_total(rho)
    vol_penalty = penalty * (np.mean(rho) - target_density) ** 2
    jax.debug.print(
        "mean(rho)={mean_rho:.3e}, J={J}, finite(J)={finite_J}",
        mean_rho=np.mean(rho),
        J=J,
        finite_J=np.all(np.isfinite(J)),
    )
    return J + vol_penalty


single_loss_and_grad = eqx.filter_value_and_grad(fem_loss_single)


def batched_loss_and_grad(models, coords, target_densities, penalties):
    return jax.vmap(single_loss_and_grad, in_axes=(0, None, 0, 0))(
        models, coords, target_densities, penalties
    )


# ---------------- Optimizer & Training ----------------
def optimisation_step(
    models, opt_states, optimizer, coords, target_densities, penalties
):
    losses, grads = batched_loss_and_grad(models, coords, target_densities, penalties)
    updates, opt_states = jax.vmap(optimizer.update)(grads, opt_states)
    models = jax.vmap(eqx.apply_updates)(models, updates)
    return models, opt_states, losses


def train_multiple_models(
    models, coords, target_densities, penalties, num_epochs=150, lr=1e-4
):
    schedule = optax.exponential_decay(
        init_value=lr,
        transition_steps=1000,
        decay_rate=0.99,
    )
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adabelief(schedule))
    opt_states = jax.vmap(lambda m: optimizer.init(eqx.filter(m, eqx.is_array)))(models)

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        models, opt_states, losses = optimisation_step(
            models, opt_states, optimizer, coords, target_densities, penalties
        )

        loss_values = np.array(losses)
        mean_loss = float(np.mean(loss_values))

        loss_str = " | ".join(
            [f"{loss_values[i]:.6f}" for i in range(len(loss_values))]
        )
        print(
            f"Epoch {epoch:03d} | mean loss = {mean_loss:.6f} | individual = [{loss_str}]"
        )

    return models, opt_states


# ---------------- Train All Models ----------------
trained_models, opt_states = train_multiple_models(
    model_batch, coords, target_densities, penalties, num_epochs=200, lr=1e-3
)

serialize_ensemble(trained_models, opt_states, ensemble_config)
