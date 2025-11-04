"""
Minimal example: Random rho -> compliance loss + grad (1000 runs)
No SIREN / no Equinox — just JAX.grad on a sigmoid-parameterized rho.
"""

import jax
import jax.numpy as np
from feax import InternalVars, create_solver
from feax import SolverOptions, zero_like_initial_guess
from feax import DirichletBCSpec, DirichletBCConfig
from feax.mesh import rectangle_mesh
from feax.topopt_toolkit import create_compliance_fn

# ---------- JAX config ----------
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)
jax.config.update("jax_debug_infs", True)
jax.config.update("jax_traceback_filtering", "off")

# ---------- Mesh / problem ----------
ele_type = "QUAD4"
Lx, Ly = 60.0, 30.0
mesh = rectangle_mesh(Nx=60, Ny=30, domain_x=Lx, domain_y=Ly)
num_elements = mesh.cells.shape[0]

# Boundary conditions
def fixed_location(point):
    return np.isclose(point[0], 0.0, atol=1e-5)

def load_location(point):
    return np.logical_and(
        np.isclose(point[0], Lx, atol=1e-5),
        np.isclose(point[1], 0.0, atol=0.1 * Ly + 1e-5),
    )

bc_config = DirichletBCConfig(
    [
        DirichletBCSpec(
            location=fixed_location,
            component="all",
            value=0.0,
        )
    ]
)

# Problem definition (same as your snippet; 2D elasticity with density)
from problems import DensityElasticityProblem
problem = DensityElasticityProblem(
    mesh=mesh,
    vec=2,
    dim=2,
    ele_type=ele_type,
    gauss_order=2,
    location_fns=[load_location],
    additional_info=(70e3, 1e-3, 0.3, 3, 1e2),  # (E, rho_min, nu, penal, load_mag) as in your code
)

# ---------- FE solver ----------
bc = bc_config.create_bc(problem)
solver_opts = SolverOptions(tol=1e-4, linear_solver="cg", use_jacobi_preconditioner=True)
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

# ---------- Compliance wrapper ----------
def J_total(rho):
    """rho is the element-wise density array (shape [num_elements])"""
    internal_vars = InternalVars(
        volume_vars=(rho,),
        surface_vars=[(traction_array,)],
    )
    sol = solver(internal_vars, initial_guess)
    return compute_compliance(sol)

# ---------- Loss + grad (no Equinox) ----------
def loss_from_z(z):
    """
    z: unconstrained parameters (R^Ne). rho = sigmoid(z) in (0,1)
    Loss = compliance + volume penalty to target vol frac 0.3
    """
    rho = jax.nn.sigmoid(z)
    jax.debug.print("mean rho: {}, min rho: {}, max rho: {}", np.mean(rho), np.min(rho), np.max(rho))
    J = J_total(rho)
    jax.debug.print("J = {}", J)
    vol_penalty = 200.0 * (np.mean(rho) - 0.3) ** 2
    return J + vol_penalty

# grad_loss_from_z = jax.jit(jax.grad(loss_from_z))
loss_from_z_jit = jax.jit(loss_from_z)

# ---------- Run 1000 random draws ----------
key = jax.random.PRNGKey(42)
for i in range(1000):
    key, sub = jax.random.split(key)
    # Random unconstrained z ~ N(0,1). Shape must match number of elements.
    z = jax.random.normal(sub, shape=(num_elements,))
    loss_val = loss_from_z_jit(z)
    # grad_val = grad_loss_from_z(z)
    # Print a tiny summary (loss + grad norm)
    # print(f"Iter {i:04d} | loss={float(loss_val):.6e} | ||grad||={float(np.linalg.norm(grad_val)):.6e}")
    print(f"Iter {i:04d} | loss={float(loss_val):.6e}")