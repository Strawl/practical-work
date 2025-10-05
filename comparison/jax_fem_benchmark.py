# Import some useful modules.
import jax.numpy as np
import pickle
import numpy as onp
import os
# import pypardiso
import scipy

# Import JAX-FEM specific modules.
from jax_fem.problem import Problem
from jax_fem.solver import solver, ad_wrapper
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import box_mesh_gmsh, get_meshio_cell_type, Mesh
from jax_fem import logger
from jax import grad
import jax

jax.config.update("jax_enable_x64", True)  # Use 64-bit precision for higher accuracy

import logging
logger.setLevel(logging.INFO)

# Material properties.
E = 70e3
nu = 0.3
mu = E/(2.*(1.+nu))
lmbda = E*nu/((1+nu)*(1-2*nu))

# Weak forms.
class LinearElasticity(Problem):
    # The function 'get_tensor_map' overrides base class method. Generally, JAX-FEM
    # solves -div(f(u_grad)) = b. Here, we have f(u_grad) = sigma.
    def get_tensor_map(self):
        def stress(u_grad):
            epsilon = 0.5 * (u_grad + u_grad.T)
            sigma = lmbda * np.trace(epsilon) * np.eye(self.dim) + 2*mu*epsilon
            return sigma
        return stress

    def set_params(self, params):
        pass

    def get_surface_maps(self):

        def surface_map(u, x):
            return np.array([0., 0., 100.])
        return [surface_map]

# Specify mesh-related information (second-order tetrahedron element).
ele_type = 'HEX8'
cell_type = get_meshio_cell_type(ele_type)
data_dir = os.path.join(os.path.dirname(__file__), 'data')
Lx, Ly, Lz = 10., 2., 2.
Nx, Ny, Nz = 25, 5, 5
meshio_mesh = box_mesh_gmsh(Nx=Nx,
                       Ny=Ny,
                       Nz=Nz,
                       domain_x=Lx,
                       domain_y=Ly,
                       domain_z=Lz,
                       data_dir=data_dir,
                       ele_type=ele_type)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

# Define boundary locations.
def left(point):
    return np.isclose(point[0], 0., atol=1e-5)

def right(point):
    return np.isclose(point[0], Lx, atol=1e-5)


# Define value function.
def zero_dirichlet_val(point):
    return 0.

# Define Dirichlet boundary values.
# This means on the 'left' side, we apply the function 'zero_dirichlet_val'
# to all components of the displacement variable u.
dirichlet_bc_info = [[left] * 3, [0, 1, 2], [zero_dirichlet_val] * 3]

# Define Neumann boundary locations.
# This means on the 'right' side, we will perform the surface integral to get
# the tractions with the function 'get_surface_maps' defined in the class 'LinearElasticity'.
location_fns = [right]

# Create an instance of the problem.
problem = LinearElasticity(mesh,
                           vec=3,
                           dim=3,
                           ele_type=ele_type,
                           dirichlet_bc_info=dirichlet_bc_info,
                           location_fns=location_fns)


import time
# solver_options = {'jax_solver': {}, 'tol':1e-8, 'precond': False}
# solver_options = {'umfpack_solver': {}, 'tol':1e-8, 'precond': True}
# solver_options = {
        # 'tol':1e-8,
        # 'petsc_solver': {
            # 'ksp_type': 'gmres',
            # 'pc_type': 'jacobi',
        # }
# }  

solver_options = {
        'tol':1e-8,
        'petsc_solver': {
            'ksp_type': 'bcgsl',
            'pc_type': 'jacobi',
        }
}  

# solver_options = {
        # 'tol':1e-8,
        # 'petsc_solver': {
            # 'ksp_type': 'bcgsl',
            # 'pc_type': 'ilu',
        # }
# }  
fwd_pred = ad_wrapper(problem,
                      solver_options=solver_options,
                      adjoint_solver_options=solver_options)

total_time = 0.0
# Define loss function using jax.numpy
def loss_fn(_=()):
    u = fwd_pred(())
    return np.mean(u[0]**2)

grad_fn = jax.grad(loss_fn)

g = grad_fn(())          # Compute gradient (backward pass)

print("Starting timed runs...")
for i in range(50):
    print(f"Run {i + 1}/50...")
    start = time.perf_counter()
    g = grad_fn(())          # Compute gradient (backward pass)
    end = time.perf_counter()

    duration = end - start
    total_time += duration
    print(f"Execution {i + 1} took {duration:.4f} seconds")

print(f"\nTotal time for 20 runs: {total_time:.4f} seconds")
print(f"Average time per run: {total_time / 50:.4f} seconds")


# vtk_path = os.path.join(data_dir, 'vtk/u.vtu')
# save_sol(
    # fe=problem.fes[0],
    # sol=sol_list[0],
    # sol_file=vtk_path,
    # point_infos=[("displacement", sol_list[0])] 
# )