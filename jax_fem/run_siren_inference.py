import jax
import jax.numpy as np
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh, rectangle_mesh
from jax_fem.utils import save_sol
import equinox as eqx
from siren import SIREN
import matplotlib.pyplot as plt

# Reload the model (must pass in a dummy instance with same architecture)


rng = jax.random.PRNGKey(42)
siren_dummy = SIREN(
    num_channels_in=2,
    num_channels_out=1,
    num_layers=4,
    num_latent_channels=64,
    omega=30.0,
    rng_key=jax.random.PRNGKey(0)   # dummy key, only used for init
)


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

ele_type = 'QUAD4'
cell_type = get_meshio_cell_type(ele_type)
Lx, Ly = 60., 30.


trained_siren_loaded = eqx.tree_deserialise_leaves("./jax_fem/trained_siren_fixed.eqx", siren_dummy)
meshio_mesh_inf = rectangle_mesh(Nx=120*10, Ny=60*10, domain_x=Lx, domain_y=Ly)
mesh_inf = Mesh(meshio_mesh_inf.points, meshio_mesh_inf.cells_dict[cell_type])
coords_inf = get_element_centroids(mesh_inf)

rho_pred_inf = jax.nn.sigmoid(trained_siren_loaded(coords_inf))

rho_img_inf = np.reshape(rho_pred_inf, (60*10, 120*10), order="F")

plt.figure(figsize=(8, 4))
plt.imshow(rho_img_inf, cmap="gray_r", origin="lower")
plt.colorbar(label="Density (rho)")
plt.title("Predicted Density Field")
plt.xlabel("x")
plt.ylabel("y")
plt.show()