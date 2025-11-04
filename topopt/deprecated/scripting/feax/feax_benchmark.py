import jax
import jax.numpy as np

from feax import Problem, InternalVars, create_solver
from feax import SolverOptions, zero_like_initial_guess
from feax import DirichletBCSpec, DirichletBCConfig
from feax.mesh import box_mesh
from feax.utils import save_sol
import os
import time

# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_log_compiles", 1)

# Problem setup
E = 70e3
T = 1e2
nu = 0.3
p = 3

class ElasticityProblem(Problem):
    def get_tensor_map(self):
        def stress(u_grad, *args):
            mu = E / (2. * (1. + nu))
            lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
            epsilon = 0.5 * (u_grad + u_grad.T)
            return lmbda * np.trace(epsilon) * np.eye(self.dim) + 2 * mu * epsilon
        return stress
    def get_surface_maps(self):
        def surface_map(u, x, traction_mag):
            return np.array([0., 0., traction_mag])
        return [surface_map]

# Mesh and boundary conditions
mesh = box_mesh(size=(10.0, 2.0, 2.0), mesh_size=0.4, element_type='HEX8')

def left(point):
    return np.isclose(point[0], 0., atol=1e-5)

def right(point):
    return np.isclose(point[0], 10.0, atol=1e-5)

bc_config = DirichletBCConfig([
    DirichletBCSpec(
        location=left,
        component='all',  # Fix x, y, z components
        value=0.0
    )
])

# Problem definition
problem = ElasticityProblem(
    mesh=mesh, vec=3, dim=3, ele_type='HEX8', gauss_order=2,
    location_fns=[right]
)


# Create boundary conditions from config
bc = bc_config.create_bc(problem)
# solver_option = SolverOptions(tol=1e-8, linear_solver="cg")
solver_option = SolverOptions(tol=1e-8, linear_solver="bicgstab", use_jacobi_preconditioner=True) # Equivalent to jax_solver
# solver_option = SolverOptions(tol=1e-8, linear_solver="bicgstab", use_jacobi_preconditioner=False) # Equivalent to jax_solver
# solver_option = SolverOptions(tol=1e-8, linear_solver="gmres", use_jacobi_preconditioner=False)
solver = create_solver(problem, bc, solver_option, iter_num=1)

initial_guess = zero_like_initial_guess(problem, bc)

def solve_forward(traction_array):
    internal_vars = InternalVars(
    volume_vars=(),
    surface_vars=[(traction_array,)]
    )
    return solver(internal_vars, initial_guess)


print(len(problem.boundary_inds_list[0]))
print(problem.fes[0].face_shape_vals.shape[1])
traction_array = InternalVars.create_uniform_surface_var(problem, T)

@jax.jit
def loss_fn(traction_array):
    sol = solve_forward(traction_array)
    return np.mean(sol**2)  # dummy scalar loss

# grad_fn = jax.jit(jax.grad(loss_fn))
grad_fn = jax.grad(loss_fn)

print("\nStarting timed loop (forward + backward)...")
total_time = 0.0
sol = solve_forward(traction_array)
g = grad_fn(traction_array)

for i in range(50):
    print(f"Run {i + 1}/50...")
    start = time.perf_counter()
    traction_array = traction_array * 1.1
    g = grad_fn(traction_array)
    jax.debug.print("shape {}", g.shape)
    jax.debug.print("Gradients: {}", g)
    end = time.perf_counter()
    duration = end - start
    total_time += duration
    print(f"Execution {i + 1} took {duration:.4f} seconds")

print(f"\nTotal time for 50 runs: {total_time:.4f} seconds")
print(f"Average time per run: {total_time / 50:.4f} seconds")
