from jax import jax
import jax.numpy as np

from feax import Mesh
from typing import  Tuple


def get_element_centroids(mesh: Mesh):
    pts = np.array(mesh.points)
    cells = np.array(mesh.cells)
    centroids = np.mean(pts[cells], axis=1)
    # normalize to [-1, 1]
    xmin, ymin = np.min(centroids, axis=0)
    xmax, ymax = np.max(centroids, axis=0)
    normalized_centroids = (centroids - np.array([xmin, ymin])) / (
        np.array([xmax - xmin, ymax - ymin])
    )
    normalized_centroids = 2.0 * normalized_centroids - 1.0
    return centroids.astype(np.float64), normalized_centroids.astype(np.float64)

def get_element_areas(mesh: Mesh):
    pts = np.asarray(mesh.points)[:, :2]
    cells = np.asarray(mesh.cells)

    if mesh.ele_type in ("TRI3", "TRI6"):
        p0 = pts[cells[:, 0]]
        p1 = pts[cells[:, 1]]
        p2 = pts[cells[:, 2]]
        areas = 0.5 * np.abs(np.cross(p1 - p0, p2 - p0))

    elif mesh.ele_type in ("QUAD4", "QUAD8"):
        p0 = pts[cells[:, 0]]
        p1 = pts[cells[:, 1]]
        p2 = pts[cells[:, 2]]
        p3 = pts[cells[:, 3]]

        tri1 = 0.5 * np.abs(np.cross(p1 - p0, p2 - p0))
        tri2 = 0.5 * np.abs(np.cross(p2 - p0, p3 - p0))
        areas = tri1 + tri2

    else:
        raise NotImplementedError(f"Area calculation not implemented for {mesh.ele_type}")

    total_area = np.sum(areas)
    return areas, total_area



def adaptive_rectangle_mesh(
    initial_size: float,
    coords: np.ndarray,
    values: np.ndarray,
    domain_x: float = 1.0,
    domain_y: float = 1.0,
    origin: Tuple[float, float] = (0.0, 0.0),
    max_depth: int = 5,
    threshold_low: float = 0.1,
    threshold_high: float = 0.9,
) -> "Mesh":
    """
    Generate an adaptive 2D rectangular mesh with QUAD4 elements,
    refined based on pointwise material values (rho).

    Parameters
    ----------
    initial_size : float
        Target size of the *coarsest* square cells (edge length).
        The domain will be partitioned into an initial structured grid
        whose cell size is close to this value.
    coords : (N, 2) array
        Coordinates of sampling points inside the domain.
        Must be in the same coordinate system as (domain_x, domain_y, origin).
    values : (N,) array
        Scalar values rho at each point in coords.
    domain_x : float, optional
        Domain length in x-direction.
    domain_y : float, optional
        Domain length in y-direction.
    origin : tuple of 2 floats, optional
        Origin (x0, y0) of the domain.
    max_depth : int, optional
        Maximum number of refinement levels (quadtree depth).
    threshold_low : float, optional
        If max(rho) < threshold_low in a cell, it is considered
        homogeneous (mostly 0) and not refined further.
    threshold_high : float, optional
        If min(rho) > threshold_high in a cell, it is considered
        homogeneous (mostly 1) and not refined further.

    Returns
    -------
    mesh : Mesh
        Mesh with QUAD4 elements.

    Notes
    -----
    - Starts from a structured grid of approximate cell size `initial_size`.
    - Each cell is recursively subdivided into 4 children if it is "mixed",
      i.e. it contains both low and high rho values.
    - Refinement stops when `max_depth` is reached or if the cell is
      homogeneous or if it contains no points.
    - This simple implementation may produce nonconforming meshes with
      hanging nodes if some cells are refined and neighbors are not.
    """

    x0, y0 = origin

    coords = np.asarray(coords, dtype=float)
    rho = np.asarray(values, dtype=float)

    if coords.shape[1] != 2:
        raise ValueError("coords must be of shape (N, 2)")
    if rho.shape[0] != coords.shape[0]:
        raise ValueError("values must have same length as coords")

    # ------------------------------------------------------------------
    # 1. Build initial coarse grid from initial_size
    # ------------------------------------------------------------------
    # number of coarse cells in each direction
    Nx = max(1, int(np.ceil(domain_x / initial_size)))
    Ny = max(1, int(np.ceil(domain_y / initial_size)))

    # actual coarse cell sizes so that the grid fits the domain exactly
    hx = domain_x / Nx
    hy = domain_y / Ny

    Npts = coords.shape[0]
    all_indices = np.arange(Npts, dtype=int)

    leaf_cells = []  # list of (xmin, xmax, ymin, ymax)

    # ------------------------------------------------------------------
    # 2. Recursive refinement
    # ------------------------------------------------------------------
    def refine_cell(xmin, xmax, ymin, ymax, depth, point_indices):
        """
        Recursively refine a cell based on rho values at the points inside it.
        """
        # Stopping condition: no points left or max depth reached
        if point_indices.size == 0 or depth >= max_depth:
            leaf_cells.append((xmin, xmax, ymin, ymax))
            return

        cell_rho = rho[point_indices]
        cell_mean = np.mean(cell_rho)

        # Homogeneous: mostly 0 or mostly 1 -> do not refine further
        if (cell_mean < threshold_low) or (cell_mean > threshold_high):
            leaf_cells.append((xmin, xmax, ymin, ymax))
            return


        # Mixed cell -> subdivide into 4 children
        xm = 0.5 * (xmin + xmax)
        ym = 0.5 * (ymin + ymax)

        xs = coords[point_indices, 0]
        ys = coords[point_indices, 1]

        # Use half-open intervals to avoid double-counting points
        # Child 0: [xmin, xm) x [ymin, ym)
        mask0 = (xs >= xmin) & (xs < xm) & (ys >= ymin) & (ys < ym)
        # Child 1: [xm, xmax] x [ymin, ym)
        mask1 = (xs >= xm) & (xs <= xmax) & (ys >= ymin) & (ys < ym)
        # Child 2: [xm, xmax] x [ym, ymax]
        mask2 = (xs >= xm) & (xs <= xmax) & (ys >= ym) & (ys <= ymax)
        # Child 3: [xmin, xm) x [ym, ymax]
        mask3 = (xs >= xmin) & (xs < xm) & (ys >= ym) & (ys <= ymax)

        idx0 = point_indices[mask0]
        idx1 = point_indices[mask1]
        idx2 = point_indices[mask2]
        idx3 = point_indices[mask3]

        refine_cell(xmin, xm, ymin, ym, depth + 1, idx0)
        refine_cell(xm,   xmax, ymin, ym, depth + 1, idx1)
        refine_cell(xm,   xmax, ym,   ymax, depth + 1, idx2)
        refine_cell(xmin, xm,   ym,   ymax, depth + 1, idx3)

    # ------------------------------------------------------------------
    # 3. Run refinement over all coarse cells
    # ------------------------------------------------------------------
    xs_all = coords[:, 0]
    ys_all = coords[:, 1]

    for i in range(Nx):
        xmin = x0 + i * hx
        xmax = x0 + (i + 1) * hx
        for j in range(Ny):
            ymin = y0 + j * hy
            ymax = y0 + (j + 1) * hy

            mask = (
                (xs_all >= xmin) & (xs_all <= xmax) &
                (ys_all >= ymin) & (ys_all <= ymax)
            )
            cell_indices = all_indices[mask]
            refine_cell(xmin, xmax, ymin, ymax, depth=0, point_indices=cell_indices)

    # ------------------------------------------------------------------
    # 4. Build points and connectivity from leaf_cells
    # ------------------------------------------------------------------
    node_map = {}   # (x,y) -> node_id
    points_list = []  # [[x,y], ...]
    cells_list = []   # [[n0,n1,n2,n3], ...]

    def get_node_id(x, y):
        # rounding to avoid floating point duplicates
        key = (round(float(x), 12), round(float(y), 12))
        if key not in node_map:
            node_map[key] = len(points_list)
            points_list.append([x, y])
        return node_map[key]

    for (xmin, xmax, ymin, ymax) in leaf_cells:
        # Node ordering: 0--1
        #                |  |
        #                3--2
        n0 = get_node_id(xmin, ymin)
        n1 = get_node_id(xmax, ymin)
        n2 = get_node_id(xmax, ymax)
        n3 = get_node_id(xmin, ymax)
        cells_list.append([n0, n1, n2, n3])

    points = np.array(points_list, dtype=float)
    cells = np.array(cells_list, dtype=np.int32)

    return Mesh(points, cells, ele_type="QUAD4")