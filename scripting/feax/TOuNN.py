"""
Minimal example: Compute compliance for a 3D elasticity problem
"""

import jax
import optax
import jax.numpy as np
from feax import Problem, InternalVars, create_solver
from feax import Mesh, SolverOptions, zero_like_initial_guess
from feax import DirichletBCSpec, DirichletBCConfig
from feax.mesh import rectangle_mesh
from feax.topopt_toolkit import create_compliance_fn
import equinox as eqx
from tqdm import tqdm
from siren import SIREN
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)  # Use 64-bit precision

# Problem setup
E0 = 70e3
E_eps = 1e-3
nu = 0.3
p = 3  # SIMP penalization parameter
T = 1e2  # Traction magnitude (fixed)

class DensityElasticityProblem(Problem):
    # def get_tensor_map(self):
        # def stress(u_grad, rho):
            # # SIMP material interpolation: E(rho) = (E0 - E_eps) * rho^p + E_eps
            # E = (E0 - E_eps) * rho**p + E_eps
            # mu = E / (2.0 * (1.0 + nu))
            # lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
            # strain = 0.5 * (u_grad + u_grad.T)
            # sigma = lam * np.trace(strain) * np.eye(self.dim) + 2.0 * mu * strain
            # return sigma
        # return stress

    def get_tensor_map(self):
        def stress(u_grad, theta):
            # Plane stress elasticity
            Emax = 70.e3
            Emin = 1e-3 * Emax
            nu = 0.3
            penal = 3.0
            E = Emin + (Emax - Emin) * theta ** penal
            epsilon = 0.5 * (u_grad + u_grad.T)
            eps11, eps22, eps12 = epsilon[0, 0], epsilon[1, 1], epsilon[0, 1]
            sig11 = E / (1 + nu) / (1 - nu) * (eps11 + nu * eps22)
            sig22 = E / (1 + nu) / (1 - nu) * (nu * eps11 + eps22)
            sig12 = E / (1 + nu) * eps12
            return np.array([[sig11, sig12], [sig12, sig22]])
        return stress
    
    def get_surface_maps(self):
        def surface_map(u, x, traction_mag):
            return np.array([0., traction_mag])
        return [surface_map]

ele_type = 'QUAD4'
Lx, Ly = 60., 30.
mesh = rectangle_mesh(Nx=60, Ny=30, domain_x=Lx, domain_y=Ly)

# Boundary conditions
def fixed_location(point): return np.isclose(point[0], 0., atol=1e-5)
def load_location(point): return np.logical_and(np.isclose(point[0], Lx, atol=1e-5),
                                               np.isclose(point[1], 0., atol=0.1*Ly + 1e-5))

bc_config = DirichletBCConfig([
    DirichletBCSpec(
        location=fixed_location,
        component='all',
        value=0.0
    )
])

problem = DensityElasticityProblem(
    mesh=mesh, vec=2, dim=2, ele_type=ele_type, gauss_order=1, location_fns=[load_location]
)

# Initial density field (all solid = 1.0)
num_elements = mesh.cells.shape[0]


# Setup FE solver
bc = bc_config.create_bc(problem)
# solver_option = SolverOptions(tol=1e-8, linear_solver="cg", use_jacobi_preconditioner=True)
solver_option = SolverOptions(tol=1e-8, linear_solver="bicgstab", use_jacobi_preconditioner=True)
solver = create_solver(problem, bc=bc, solver_options=solver_option, adjoint_solver_options=solver_option, iter_num=1)

initial_guess = zero_like_initial_guess(problem, bc)
traction_array = InternalVars.create_uniform_surface_var(problem, T)

# Compliance function
compute_compliance = create_compliance_fn(problem, surface_load_params=T)

# Build internal variables


def get_element_centroids(mesh):
    pts = np.array(mesh.points)
    cells = np.array(mesh.cells)
    centroids = np.mean(pts[cells], axis=1)  # (num_cells, 2)

    # normalize to [-1, 1] for SIREN stability
    xmin, ymin = np.min(centroids, axis=0)
    xmax, ymax = np.max(centroids, axis=0)
    centroids = (centroids - np.array([xmin, ymin])) / (np.array([xmax - xmin, ymax - ymin]))
    centroids = 2.0 * centroids - 1.0

    return centroids.astype(np.float32)

def J_total(params):
    global last_solution
    internal_vars = InternalVars(
        volume_vars=(params,),
        surface_vars=[(traction_array,)]
    )
    sol = solver(internal_vars, initial_guess)
    last_solution = sol 
    return compute_compliance(sol)

@eqx.filter_jit
def fem_loss(model, coords):
    rho = jax.nn.sigmoid(model(coords))

    # def _plot_callback(rho_host):
        # rho_img = np.reshape(rho_host, (30, 60), order="F")
        # plt.figure(figsize=(8, 4))
        # plt.imshow(rho_img, cmap="gray_r", origin="lower")
        # plt.colorbar(label="Density (rho)")
        # plt.title("Predicted Density Field")
        # plt.xlabel("x")
        # plt.ylabel("y")
        # plt.show()

    # jax.debug.callback(_plot_callback, rho)

    J = J_total(rho)
    vol_penalty = 200 * (np.mean(rho) - 0.3) ** 2
    total_loss = J + vol_penalty
    return total_loss 

loss_and_grad = eqx.filter_value_and_grad(fem_loss)

def optimisation_step(model, optimiser, opt_state, coords):
    loss, grads = loss_and_grad(model, coords)
    updates, opt_state = optimiser.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

def train_siren(model, coords, num_epochs=500, lr=1e-3, save_dir="./checkpoints"):
    optimiser = optax.adam(lr)
    opt_state = optimiser.init(eqx.filter(model, eqx.is_array))

    # opt_state = eqx.tree_deserialise_leaves(
        # f"./feax/opt_state_epoch_85.eqx", opt_state
    # )

    prev_loss = float("inf")

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        model, opt_state, loss = optimisation_step(model, optimiser, opt_state, coords)
        loss_val = float(loss)

        print(f"Epoch {epoch}, loss = {loss_val:.6e}")

        # Stop if loss increases compared to previous step
        if loss_val > prev_loss:
            print(f"⚠️ Loss increased at epoch {epoch}: {loss_val:.6e} (prev {prev_loss:.6e})")
            eqx.tree_serialise_leaves(f"{save_dir}/siren_epoch_{epoch}_loss_increase.eqx", model)
            eqx.tree_serialise_leaves(f"{save_dir}/opt_state_epoch_{epoch}_loss_increase.eqx", opt_state)
            break

        prev_loss = loss_val

    return model

coords = get_element_centroids(mesh)

rng = jax.random.PRNGKey(42)
siren = SIREN(
    num_channels_in=2,
    num_channels_out=1,
    num_layers=4,
    num_latent_channels=64,
    omega=30.0,
    rng_key=rng
)

#with jax.profiler.trace("./profile-data"):
# siren = eqx.tree_deserialise_leaves(f"./feax/siren_epoch_85.eqx", siren)
trained_siren = train_siren(siren, coords, num_epochs=200, lr=1e-3)
eqx.tree_serialise_leaves("./feax/trained_siren.eqx", trained_siren)
