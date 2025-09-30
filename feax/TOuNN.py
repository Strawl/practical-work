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

# jax.config.update("jax_enable_x64", True)  # Use 64-bit precision
print(jax.devices())
print(jax.default_backend())  

# Problem setup
E0 = 70e3
E_eps = 1e-3
nu = 0.3
p = 3  # SIMP penalization parameter
T = 1e2  # Traction magnitude (fixed)

class DensityElasticityProblem(Problem):
    def get_tensor_map(self):
        def stress(u_grad, rho):
            # SIMP material interpolation: E(rho) = (E0 - E_eps) * rho^p + E_eps
            E = (E0 - E_eps) * rho**p + E_eps
            mu = E / (2.0 * (1.0 + nu))
            lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
            strain = 0.5 * (u_grad + u_grad.T)
            sigma = lam * np.trace(strain) * np.eye(self.dim) + 2.0 * mu * strain
            return sigma
        return stress
    
    def get_surface_maps(self):
        def surface_map(u, x, *args):
            return np.array([0., 100.])
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
solver_option = SolverOptions(tol=1e-8, linear_solver="cg", use_jacobi_preconditioner=True)
solver = create_solver(problem, bc, solver_option, iter_num=1)
initial_guess = zero_like_initial_guess(problem, bc)

# Compliance function
compute_compliance = create_compliance_fn(problem, surface_load_params=T)

# Build internal variables

@jax.jit
def J_total(params):
    internal_vars = InternalVars(
        volume_vars=(params,),
        surface_vars=[]
    )
    sol = solver(internal_vars, initial_guess)
    return compute_compliance(sol)

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

@eqx.filter_jit
def fem_loss(model, coords, vf=np.array(0.3), penalty=np.array(200.0)):
    rho = jax.nn.sigmoid(model(coords))
    J = J_total(rho)
    vol_penalty = penalty * (np.mean(rho) - vf) ** 2
    return J + vol_penalty

loss_and_grad = eqx.filter_value_and_grad(fem_loss)

@eqx.filter_jit
def optimisation_step(model, optimiser, opt_state, coords):
    loss, grads = loss_and_grad(model, coords)
    updates, opt_state = optimiser.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

def train_siren(model, coords, num_epochs=500, lr=1e-3):
    optimiser = optax.adam(lr)
    opt_state = optimiser.init(eqx.filter(model, eqx.is_array))

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        model, opt_state, loss = optimisation_step(model, optimiser, opt_state, coords)

        if epoch % 5 == 0:
            print(f"Epoch {epoch}, loss = {float(loss)}")

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

trained_siren = train_siren(siren, coords, num_epochs=500, lr=1e-3)
eqx.tree_serialise_leaves("./feax/trained_siren.eqx", trained_siren)
