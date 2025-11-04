import jax.numpy as np
from feax import Mesh

def get_element_centroids(mesh: Mesh):
    pts = np.array(mesh.points)
    cells = np.array(mesh.cells)
    centroids = np.mean(pts[cells], axis=1)
    # normalize to [-1, 1]
    xmin, ymin = np.min(centroids, axis=0)
    xmax, ymax = np.max(centroids, axis=0)
    centroids = (centroids - np.array([xmin, ymin])) / (np.array([xmax - xmin, ymax - ymin]))
    centroids = 2.0 * centroids - 1.0
    return centroids.astype(np.float64)