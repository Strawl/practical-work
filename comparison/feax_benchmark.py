"""
Linear elasticity example with SIMP-based material interpolation.
Demonstrates density-dependent material properties using the SIMP (Solid Isotropic Material with Penalization) method.
"""

import jax
import jax.numpy as np
from feax import Problem, InternalVars, create_solver
from feax import Mesh, SolverOptions, zero_like_initial_guess
from feax import DirichletBCSpec, DirichletBCConfig
from feax.mesh import box_mesh
from feax.utils import save_sol
import os
import time
jax.config.update("jax_enable_x64", True)  # Use 64-bit precision for higher accuracy

# Problem setup
E = 70e3
nu = 0.3
mu = E/(2.*(1.+nu))
lmbda = E*nu/((1+nu)*(1-2*nu))
T = 1e2

class LinearElasticity(Problem):
    def get_tensor_map(self):
        def stress(u_grad):
            epsilon = 0.5 * (u_grad + u_grad.T)
            return lmbda * np.trace(epsilon) * np.eye(self.dim) + 2 * mu * epsilon
        return stress
    
    def get_surface_maps(self):
        def surface_map(u, x, traction_mag):
            return np.array([0., 0., traction_mag])
        return [surface_map]

# Create mesh and problem
Lx, Ly, Lz = 10., 2., 2.
Nx, Ny, Nz = 25, 5, 5
mesh = box_mesh(Nx=Nx,
                       Ny=Ny,
                       Nz=Nz,
                       domain_x=Lx,
                       domain_y=Ly,
                       domain_z=Lz)

def left(point):
    return np.isclose(point[0], 0., atol=1e-5)

def right(point):
    return np.isclose(point[0], Lx, atol=1e-5)

# Create boundary conditions using the new dataclass approach
bc_config = DirichletBCConfig([
    # Fix left boundary completely (all components to zero)
    DirichletBCSpec(
        location=left,
        component='all',  # Fix x, y, z components
        value=0.0
    )
])

problem = LinearElasticity(
    mesh=mesh, vec=3, dim=3, ele_type='HEX8', gauss_order=2,
    location_fns=[right]
)

# Create InternalVars separately
traction_array = InternalVars.create_uniform_surface_var(problem, T)

# Create boundary conditions from config
bc = bc_config.create_bc(problem)
# solver_option = SolverOptions(tol=1e-8, linear_solver="cg")
# solver_option = SolverOptions(tol=1e-8, linear_solver="bicgstab", use_jacobi_preconditioner=False) # Equivalent to jax_solver
solver_option = SolverOptions(tol=1e-8, linear_solver="bicgstab", use_jacobi_preconditioner=True) # Equivalent to jax_solver
# solver_option = SolverOptions(tol=1e-8, linear_solver="gmres", use_jacobi_preconditioner=False)
solver = create_solver(problem, bc, solver_option, iter_num=1)

initial_guess = zero_like_initial_guess(problem, bc)

# @jax.jit
def solve_forward():
    internal_vars = InternalVars(
    surface_vars=[(traction_array,)]
    )
    return solver(internal_vars, initial_guess)


def loss_fn(_=()):
    sol = solve_forward()
    sol_unflat = problem.unflatten_fn_sol_list(sol)
    displacement = sol_unflat[0]
    return np.mean(displacement**2)  # dummy scalar loss

# grad_fn = jax.jit(jax.grad(loss_fn))
grad_fn = jax.grad(loss_fn)

# Benchmark backward (forward + adjoint) timing
print("\nStarting timed loop (forward + backward)...")
total_time = 0.0
g = grad_fn(())

for i in range(50):
    print(f"Run {i + 1}/20...")
    start = time.perf_counter()
    g = grad_fn(())
    end = time.perf_counter()
    duration = end - start
    total_time += duration
    print(f"Execution {i + 1} took {duration:.4f} seconds")

print(f"\nTotal time for 20 runs: {total_time:.4f} seconds")
print(f"Average time per run: {total_time / 50:.4f} seconds")

# numdofs = problem.num_total_dofs_all_vars
# sol_unflat = problem.unflatten_fn_sol_list(sol)

# displacement = sol_unflat[0]

# data_dir = os.path.join(os.path.dirname(__file__), 'data')
# os.makedirs(os.path.join(data_dir, 'vtk'), exist_ok=True)
# vtk_path = os.path.join(data_dir, 'vtk/u2.vtu')

# save_sol(
    # mesh=mesh,
    # sol_file=vtk_path,
    # point_infos=[("displacement", displacement)])