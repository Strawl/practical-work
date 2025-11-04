import jax
import jax.numpy as np
import equinox as eqx
import matplotlib.pyplot as plt
from feax.mesh import rectangle_mesh
from jax.nn import sigmoid
from topopt.siren import SIREN
from pathlib import Path


# ------------------------------
# 1. Setup mesh and helper utils
# ------------------------------
def get_element_centroids(mesh):
    """Compute normalized element centroids for SIREN inference."""
    pts = np.array(mesh.points)
    cells = np.array(mesh.cells)
    centroids = np.mean(pts[cells], axis=1)  # (num_cells, 2)

    # normalize to [-1, 1] for SIREN stability
    xmin, ymin = np.min(centroids, axis=0)
    xmax, ymax = np.max(centroids, axis=0)
    centroids = (centroids - np.array([xmin, ymin])) / (np.array([xmax - xmin, ymax - ymin]))
    centroids = 2.0 * centroids - 1.0

    return centroids.astype(np.float32)


# ------------------------------
# 2. Load trained model
# ------------------------------
run_dir = Path("outputs") / "2025-10-30_18-53-31"  # e.g., 2025-10-30_15-00-00
model_path = run_dir / "trained_siren.eqx"

rng = jax.random.PRNGKey(0)
siren_dummy = SIREN(
    num_channels_in=2,
    num_channels_out=1,
    num_layers=4,          # must match training
    num_latent_channels=64,
    omega=30.0,
    rng_key=rng
)

model = eqx.tree_deserialise_leaves(model_path, siren_dummy)


# ------------------------------
# 3. Prepare mesh for inference
# ------------------------------
Lx, Ly = 60., 30.
Nx, Ny = 60*5, 30*5
mesh = rectangle_mesh(Nx=Nx, Ny=Ny, domain_x=Lx, domain_y=Ly)
coords = get_element_centroids(mesh)


# ------------------------------
# 4. Run model prediction
# ------------------------------
rho_pred = sigmoid(model(coords))  # (num_elements, 1)
rho_pred = np.reshape(rho_pred, (Ny, Nx), order="F")

print(f"Predicted rho range: [{rho_pred.min():.3f}, {rho_pred.max():.3f}]")

# ------------------------------
# 5. Plot and save visualization
# ------------------------------
plt.figure(figsize=(8, 4))
plt.imshow(rho_pred, cmap="gray_r", origin="lower")
plt.colorbar(label="Density (ρ)")
plt.title("Predicted Density Field")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()

out_path = run_dir / "rho_pred.png"
plt.savefig(out_path, dpi=300)
plt.show()

print(f"Saved visualization to {out_path}")