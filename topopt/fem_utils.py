import math
from typing import Tuple

import feax.flat as flat
import jax.numpy as jnp
from feax.gene import create_compliance_fn, create_volume_fn

from feax import (
    DirichletBCConfig,
    DirichletBCSpec,
    InternalVars,
    Mesh,
    SolverOptions,
    create_solver,
    zero_like_initial_guess,
)
from topopt.problems import DensityElasticityProblem


def get_element_geometry(mesh):
    """
    Computes:
      - raw centroids
      - centroids normalized to [-1, 1] using centroid min/max (your current scheme)
      - per-element bbox sizes in raw space
      - per-element bbox sizes converted into the SAME normalized space
      - num_elements

    Returns JAX arrays (except num_elements).
    """
    pts = jnp.asarray(mesh.points)[:, :2]
    cells = jnp.asarray(mesh.cells, dtype=jnp.int32)

    elem_pts = pts[cells]
    centroids = elem_pts.mean(axis=1)

    cmin = centroids.min(axis=0)
    cmax = centroids.max(axis=0)
    crange = cmax - cmin

    crange = jnp.where(crange == 0, 1.0, crange)

    centroids_scaled = 2.0 * ((centroids - cmin) / crange) - 1.0

    mins = elem_pts.min(axis=1)
    maxs = elem_pts.max(axis=1)
    sizes_raw = maxs - mins
    dx_raw = sizes_raw[:, 0]
    dy_raw = sizes_raw[:, 1]

    dx_scaled = 2.0 * dx_raw / crange[0]
    dy_scaled = 2.0 * dy_raw / crange[1]

    num_elements = cells.shape[0]

    return {
        "centroids": jnp.asarray(centroids, dtype=jnp.float64),
        "centroids_scaled": jnp.asarray(centroids_scaled, dtype=jnp.float64),
        "dx_raw": jnp.asarray(dx_raw),
        "dy_raw": jnp.asarray(dy_raw),
        "dx_scaled": jnp.asarray(dx_scaled),
        "dy_scaled": jnp.asarray(dy_scaled),
        "centroid_min": jnp.asarray(cmin),
        "centroid_range": jnp.asarray(crange),
        "num_elements": int(num_elements),
    }


def create_objective_functions(
    mesh,
    fixed_location,
    load_location,
    target_fraction=None,
    ele_type="QUAD4",
    E0=70e3,
    E_eps=7,
    nu=0.3,
    p=3,
    T=1e2,
    gauss_order=2,
    iter_num=1,
    check_convergence=False,
    verbose=False,
    radius: float = 0,
    fwd_linear_solver: str = "bicgstab",  # "gmres" or "spsolve"
    bwd_linear_solver: str = "bicgstab",  # "gmres" or "spsolve"
):
    bc_config = DirichletBCConfig(
        [DirichletBCSpec(location=fixed_location, component="all", value=0.0)]
    )

    problem = DensityElasticityProblem(
        mesh=mesh,
        vec=2,
        dim=2,
        ele_type=ele_type,
        gauss_order=gauss_order,
        location_fns=[load_location],
        additional_info=(E0, E_eps, nu, p, T),
    )

    bc = bc_config.create_bc(problem)

    solver_options = SolverOptions(
        tol=1e-8,
        linear_solver_tol=1e-10,
        linear_solver_atol=1e-10,
        linear_solver=fwd_linear_solver,
        use_jacobi_preconditioner=True,
        check_convergence=check_convergence,
        verbose=verbose,
        linear_solver_maxiter=10000,
    )

    adjoint_solver_options = SolverOptions(
        tol=1e-8,
        linear_solver_tol=1e-10,
        linear_solver_atol=1e-10,
        linear_solver=bwd_linear_solver,
        use_jacobi_preconditioner=True,
        check_convergence=check_convergence,
        verbose=verbose,
        linear_solver_maxiter=10000,
    )

    solver = create_solver(
        problem,
        bc=bc,
        solver_options=solver_options,
        adjoint_solver_options=adjoint_solver_options,
        iter_num=iter_num,
    )

    initial_guess = zero_like_initial_guess(problem, bc)

    traction_array = InternalVars.create_uniform_surface_var(problem, T)
    compute_compliance = create_compliance_fn(problem, surface_load_params=problem.T)

    if radius <= 0:
        def filter_fn(rho):
            return rho
    else:
        filter_fn = flat.filters.create_helmholtz_filter(mesh, radius)

    def solve_forward(rho):
        """Compute compliance for given node-based density field."""
        internal_vars = InternalVars(
            volume_vars=(rho,),
            surface_vars=[(traction_array,)],
        )
        sol = solver(internal_vars, initial_guess)
        return compute_compliance(sol)

    volume_fn = create_volume_fn(problem)

    def evaluate_volume(rho):
        """Compute volume fraction for given node-based density field."""
        return volume_fn(rho)

    rho_init = None
    if target_fraction:
        rho_init = InternalVars.create_node_var(problem, target_fraction)

    return solve_forward, evaluate_volume, filter_fn, rho_init, mesh.points.shape[0]


def adaptive_rectangle_mesh_new(
    initial_size: float,
    coords,
    values,
    domain_x: float = 1.0,
    domain_y: float = 1.0,
    origin: Tuple[float, float] = (0.0, 0.0),
    max_depth: int = 5,
    threshold_low: float = 0.1,
    threshold_high: float = 0.9,
):
    x0, y0 = origin

    # Convert to JAX arrays
    coords = jnp.asarray(coords, dtype=jnp.float32)
    rho = jnp.asarray(values, dtype=jnp.float32)

    # Check shapes
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must be of shape (N, 2)")
    if rho.shape[0] != coords.shape[0]:
        raise ValueError("values must have same length as coords")

    # -------------------------------------------------------------
    # 1. Coarse grid + finest grid
    # -------------------------------------------------------------
    Nx0 = max(1, int(math.ceil(domain_x / float(initial_size))))
    Ny0 = max(1, int(math.ceil(domain_y / float(initial_size))))

    hx0 = domain_x / Nx0
    hy0 = domain_y / Ny0

    factor = 1 << max_depth  # 2**max_depth
    Nx_fine = Nx0 * factor
    Ny_fine = Ny0 * factor

    hx_fine = hx0 / factor
    hy_fine = hy0 / factor

    # -------------------------------------------------------------
    # 2. Bin points to finest cells
    # -------------------------------------------------------------
    xs = coords[:, 0]
    ys = coords[:, 1]

    ix_f = jnp.floor((xs - x0) / hx_fine).astype(jnp.int32)
    iy_f = jnp.floor((ys - y0) / hy_fine).astype(jnp.int32)

    ix_f = jnp.clip(ix_f, 0, Nx_fine - 1)
    iy_f = jnp.clip(iy_f, 0, Ny_fine - 1)

    # 1D ID
    cell_ids = ix_f * Ny_fine + iy_f
    n_cells_fine = Nx_fine * Ny_fine

    # Init aggregates
    min_rho_flat = jnp.full((n_cells_fine,), jnp.inf, dtype=jnp.float32)
    max_rho_flat = jnp.full((n_cells_fine,), -jnp.inf, dtype=jnp.float32)
    cnt_flat = jnp.zeros((n_cells_fine,), dtype=jnp.int32)

    # Scatter reduce
    min_rho_flat = min_rho_flat.at[cell_ids].min(rho)
    max_rho_flat = max_rho_flat.at[cell_ids].max(rho)
    cnt_flat = cnt_flat.at[cell_ids].add(1)

    # Finest level 2D arrays
    min_rho_fine = min_rho_flat.reshape(Nx_fine, Ny_fine)
    max_rho_fine = max_rho_flat.reshape(Nx_fine, Ny_fine)
    cnt_fine = cnt_flat.reshape(Nx_fine, Ny_fine)

    # -------------------------------------------------------------
    # 3. Bottom-up construction of per-level aggregates
    # -------------------------------------------------------------
    min_levels = [None] * (max_depth + 1)
    max_levels = [None] * (max_depth + 1)
    cnt_levels = [None] * (max_depth + 1)

    min_levels[max_depth] = min_rho_fine
    max_levels[max_depth] = max_rho_fine
    cnt_levels[max_depth] = cnt_fine

    Nx_level = Nx_fine
    Ny_level = Ny_fine

    for d in range(max_depth - 1, -1, -1):
        Nx_parent = Nx_level // 2
        Ny_parent = Ny_level // 2

        min_child = min_levels[d + 1]
        max_child = max_levels[d + 1]
        cnt_child = cnt_levels[d + 1]

        # Shape (Nx_parent,2,Ny_parent,2)
        min_parent = min_child.reshape(Nx_parent, 2, Ny_parent, 2).min(axis=(1, 3))
        max_parent = max_child.reshape(Nx_parent, 2, Ny_parent, 2).max(axis=(1, 3))
        cnt_parent = cnt_child.reshape(Nx_parent, 2, Ny_parent, 2).sum(axis=(1, 3))

        min_levels[d] = min_parent
        max_levels[d] = max_parent
        cnt_levels[d] = cnt_parent

        Nx_level = Nx_parent
        Ny_level = Ny_parent

    assert Nx_level == Nx0 and Ny_level == Ny0

    # -------------------------------------------------------------
    # 4. Top-down AMR to get leaf cells
    # -------------------------------------------------------------
    stack = [(0, i, j) for i in range(Nx0) for j in range(Ny0)]
    leaves = []

    while stack:
        d, i, j = stack.pop()

        min_d = float(min_levels[d][i, j])
        max_d = float(max_levels[d][i, j])
        cnt_d = int(cnt_levels[d][i, j])

        if cnt_d == 0:
            leaves.append((d, i, j))
            continue

        if d >= max_depth:
            leaves.append((d, i, j))
            continue

        if (max_d < threshold_low) or (min_d > threshold_high):
            leaves.append((d, i, j))
            continue

        # refine
        child = d + 1
        ci0 = 2 * i
        cj0 = 2 * j
        stack.append((child, ci0, cj0))
        stack.append((child, ci0 + 1, cj0))
        stack.append((child, ci0 + 1, cj0 + 1))
        stack.append((child, ci0, cj0 + 1))

    # -------------------------------------------------------------
    # 5. Build mesh nodes and connectivity
    # -------------------------------------------------------------
    x_nodes = jnp.linspace(x0, x0 + domain_x, Nx_fine + 1)
    y_nodes = jnp.linspace(y0, y0 + domain_y, Ny_fine + 1)

    Xg, Yg = jnp.meshgrid(x_nodes, y_nodes, indexing="ij")
    points = jnp.stack([Xg.ravel(), Yg.ravel()], axis=1)

    n_cells = len(leaves)
    cells = jnp.zeros((n_cells, 4), dtype=jnp.int32)

    n_y_nodes = Ny_fine + 1

    # Build connectivity with .at[]
    for e, (d, i, j) in enumerate(leaves):
        step = 1 << (max_depth - d)

        ix0 = i * step
        ix1 = (i + 1) * step
        iy0 = j * step
        iy1 = (j + 1) * step

        n0 = ix0 * n_y_nodes + iy0
        n1 = ix1 * n_y_nodes + iy0
        n2 = ix1 * n_y_nodes + iy1
        n3 = ix0 * n_y_nodes + iy1

        cells = cells.at[e, :].set(jnp.array([n0, n1, n2, n3], dtype=jnp.int32))

    # -------------------------------------------------------------
    # 6. Return mesh
    # -------------------------------------------------------------
    return Mesh(points, cells, ele_type="QUAD4")


def adaptive_rectangle_mesh(
    initial_size: float,
    coords: jnp.ndarray,
    values: jnp.ndarray,
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

    coords = jnp.asarray(coords, dtype=float)
    rho = jnp.asarray(values, dtype=float)

    if coords.shape[1] != 2:
        raise ValueError("coords must be of shape (N, 2)")
    if rho.shape[0] != coords.shape[0]:
        raise ValueError("values must have same length as coords")

    # ------------------------------------------------------------------
    # 1. Build initial coarse grid from initial_size
    # ------------------------------------------------------------------
    # number of coarse cells in each direction
    Nx = max(1, int(jnp.ceil(domain_x / initial_size)))
    Ny = max(1, int(jnp.ceil(domain_y / initial_size)))

    # actual coarse cell sizes so that the grid fits the domain exactly
    hx = domain_x / Nx
    hy = domain_y / Ny

    jnpts = coords.shape[0]
    all_indices = jnp.arange(jnpts, dtype=int)

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
        cell_mean = jnp.mean(cell_rho)
        # maximum = jnp.max(cell_rho)
        # minimum = jnp.min(cell_rho)

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
        refine_cell(xm, xmax, ymin, ym, depth + 1, idx1)
        refine_cell(xm, xmax, ym, ymax, depth + 1, idx2)
        refine_cell(xmin, xm, ym, ymax, depth + 1, idx3)

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
                (xs_all >= xmin)
                & (xs_all <= xmax)
                & (ys_all >= ymin)
                & (ys_all <= ymax)
            )
            cell_indices = all_indices[mask]
            refine_cell(xmin, xmax, ymin, ymax, depth=0, point_indices=cell_indices)

    # ------------------------------------------------------------------
    # 4. Build points and connectivity from leaf_cells
    # ------------------------------------------------------------------
    node_map = {}  # (x,y) -> node_id
    points_list = []  # [[x,y], ...]
    cells_list = []  # [[n0,n1,n2,n3], ...]

    def get_node_id(x, y):
        # rounding to avoid floating point duplicates
        key = (round(float(x), 12), round(float(y), 12))
        if key not in node_map:
            node_map[key] = len(points_list)
            points_list.append([x, y])
        return node_map[key]

    for xmin, xmax, ymin, ymax in leaf_cells:
        # Node ordering: 0--1
        #                |  |
        #                3--2
        n0 = get_node_id(xmin, ymin)
        n1 = get_node_id(xmax, ymin)
        n2 = get_node_id(xmax, ymax)
        n3 = get_node_id(xmin, ymax)
        cells_list.append([n0, n1, n2, n3])

    points = jnp.array(points_list, dtype=float)
    cells = jnp.array(cells_list, dtype=jnp.int32)

    return Mesh(points, cells, ele_type="QUAD4")
