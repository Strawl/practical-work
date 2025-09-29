import jax
import optax
import jax.numpy as np
from jax_fem.problem import Problem
from jax_fem.solver import solver, ad_wrapper
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh, rectangle_mesh
from jax_fem.utils import save_sol
import equinox as eqx
from tqdm import tqdm
from siren import SIREN
import matplotlib.pyplot as plt
import logging

logging.getLogger("jax_fem").setLevel(logging.WARNING)



# -----------------------------
# Define elasticity problem
# -----------------------------
class Elasticity(Problem):
    def custom_init(self):
        self.fe = self.fes[0]
        self.fe.flex_inds = np.arange(len(self.fe.cells))

    def get_tensor_map(self):
        def stress(u_grad, theta):
            # Plane stress elasticity
            Emax = 70.e3
            Emin = 1e-3 * Emax
            nu = 0.3
            penal = 3.0
            E = Emin + (Emax - Emin) * theta[0] ** penal
            epsilon = 0.5 * (u_grad + u_grad.T)
            eps11, eps22, eps12 = epsilon[0, 0], epsilon[1, 1], epsilon[0, 1]
            sig11 = E / (1 + nu) / (1 - nu) * (eps11 + nu * eps22)
            sig22 = E / (1 + nu) / (1 - nu) * (nu * eps11 + eps22)
            sig12 = E / (1 + nu) * eps12
            return np.array([[sig11, sig12], [sig12, sig22]])
        return stress

    def set_params(self, params):
        full_params = np.ones((self.fe.num_cells, params.shape[1]))
        full_params = full_params.at[self.fe.flex_inds].set(params)
        thetas = np.repeat(full_params[:, None, :], self.fe.num_quads, axis=1)
        self.full_params = full_params
        self.internal_vars = [thetas]

    def compute_compliance(self, sol):
        boundary_inds = self.boundary_inds_list[0]
        _, nanson_scale = self.fe.get_face_shape_grads(boundary_inds)
        u_face = sol[self.fe.cells][boundary_inds[:, 0]][:, None, :, :] \
               * self.fe.face_shape_vals[boundary_inds[:, 1]][:, :, :, None]
        u_face = np.sum(u_face, axis=2)
        subset_quad_points = self.physical_surface_quad_points[0]
        neumann_fn = self.get_surface_maps()[0]
        traction = -jax.vmap(jax.vmap(neumann_fn))(u_face, subset_quad_points)
        return np.sum(traction * u_face * nanson_scale[:, :, None])

    def get_surface_maps(self):
        def surface_map(u, x):
            return np.array([0., 100.])  # load
        return [surface_map]


# -----------------------------
# Mesh and boundary conditions
# -----------------------------
ele_type = 'QUAD4'
cell_type = get_meshio_cell_type(ele_type)
Lx, Ly = 60., 30.
meshio_mesh = rectangle_mesh(Nx=240, Ny=120, domain_x=Lx, domain_y=Ly)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

def fixed_location(point): return np.isclose(point[0], 0., atol=1e-5)
def load_location(point): return np.logical_and(np.isclose(point[0], Lx, atol=1e-5),
                                               np.isclose(point[1], 0., atol=0.1*Ly + 1e-5))
def dirichlet_val(point): return 0.

dirichlet_bc_info = [[fixed_location]*2, [0, 1], [dirichlet_val]*2]
location_fns = [load_location]

problem = Elasticity(mesh, vec=2, dim=2, ele_type=ele_type,
                     dirichlet_bc_info=dirichlet_bc_info,
                     location_fns=location_fns)

# Differentiable solver
fwd_pred = ad_wrapper(problem,
                      solver_options={'umfpack_solver': {}},
                      adjoint_solver_options={'umfpack_solver': {}})

def J_total(params):
    sol_list = fwd_pred(params)
    return problem.compute_compliance(sol_list[0])

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

def fem_loss(model, coords, vf=0.3, penalty=200.0):
    rho = jax.nn.sigmoid(model(coords))
    J = J_total(rho)
    vol_penalty = penalty * (np.mean(rho) - vf) ** 2
    return J + vol_penalty

loss_and_grad = eqx.filter_value_and_grad(fem_loss)

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

        if epoch % 50 == 0:
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

trained_siren = train_siren(siren, coords, num_epochs=1000, lr=1e-3)
eqx.tree_serialise_leaves("trained_siren.eqx", trained_siren)