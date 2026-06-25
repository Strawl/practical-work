#!/usr/bin/env python3
"""
Evaluate compliance of a TOuNN-exported .vtu file using feax.

This script reads a .vtu file (exported from TOuNN with export_density_vtu.py),
creates a matching feax mesh, and evaluates the structural compliance with
hardcoded material properties and standard cantilever boundary conditions.

Usage:
    python -m topopt.evaluate_vtu path/to/file.vtu [--bc cantilever_corner] [--penal 3.0]

Example:
    python -m topopt.evaluate_vtu results/tounn_density_res15.vtu
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import jax.numpy as jnp
import numpy as np
import pyvista as pv

from feax import (
    DirichletBCConfig,
    InternalVars,
    Mesh,
    create_solver,
    zero_like_initial_guess,
)
from feax.gene import create_dynamic_compliance_fn, create_volume_fn

from topopt.bc import (
    canonicalize_neumann_boundary_conditions,
    make_dirichlet_boundary_conditions,
    make_neumann_boundary_location,
    make_neumann_surface_var_fn,
)
from topopt.fem_utils import create_surface_vars
from topopt.problems import PlaneStressElasticityProblem
from topopt.solver_config import build_solver_setup


# =============================================================================
# Hardcoded material properties (matching TOuNN defaults / standard values)
# =============================================================================

# Standard structural steel-like properties
E0 = 1.0  # Young's modulus of solid material (normalized)
Emin = 1e-6  # Young's modulus of void material (normalized)
NU = 0.3  # Poisson's ratio
PENAL = 3.0  # SIMP penalization exponent
TRACTION = 1.0  # Traction magnitude (normalized)
POINT_LOAD_MAGNITUDE = 1.0  # TOuNN default nodal force magnitude


# =============================================================================
# VTU parsing
# =============================================================================

def parse_vtu_raw(vtu_path: str | Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    Parse a TOuNN-exported VTU file and return (points, cells, density, nx, ny).

    The VTU uses VTK_QUAD cells. Density may be stored either as:
    - cell_data["density"] for cell-centered fields
    - point_data["density"] for node-based fields

    Node-based densities are converted to cell-centered values by averaging
    each element's corner-node densities.

    Returns
    -------
    points : np.ndarray, shape (n_points, 2)
        Node coordinates in 2D (z stripped).
    cells : np.ndarray, shape (n_cells, 4)
        QUAD4 connectivity (0-based indexing).
    density : np.ndarray, shape (n_cells,)
        Cell-centered density values.
    nx : int
        Number of elements in x-direction.
    ny : int
        Number of elements in y-direction.
    """
    vtu_path = Path(vtu_path)
    if not vtu_path.exists():
        raise FileNotFoundError(f"VTU file not found: {vtu_path}")

    # PyVista can read VTK XML directly
    mesh = pv.read(str(vtu_path))

    # Extract points - strip z to make it 2D
    points_3d = mesh.points  # shape (n_points, 3)
    points = np.array(points_3d[:, :2], dtype=np.float64)

    # Check z is zero (2D mesh)
    if not np.allclose(points_3d[:, 2], 0.0):
        raise ValueError("VTU mesh is not planar (z != 0). Expected 2D mesh.")

    # Extract cells
    cell_types = mesh.celltypes
    unique_types = np.unique(cell_types)
    if len(unique_types) > 1 or unique_types[0] != 9:  # 9 = VTK_QUAD
        raise ValueError(
            f"Expected all cells to be VTK_QUAD (type 9), got types: {unique_types}"
        )

    # Get connectivity
    cells_flat = mesh.cells
    n_cells = mesh.n_cells
    cells = cells_flat.reshape(n_cells, 5)[:, 1:].astype(np.int32)

    # Extract density, accepting either cell-centered or node-based storage.
    if "density" in mesh.cell_data:
        density = np.array(mesh.cell_data["density"], dtype=np.float64).ravel()
        if len(density) != n_cells:
            raise ValueError(
                f"Cell density array length ({len(density)}) != number of cells ({n_cells})"
            )
    elif "density" in mesh.point_data:
        point_density = np.array(mesh.point_data["density"], dtype=np.float64).ravel()
        if len(point_density) != points.shape[0]:
            raise ValueError(
                f"Point density array length ({len(point_density)}) != number of points ({points.shape[0]})"
            )
        density = point_density[cells].mean(axis=1)
    else:
        raise ValueError(
            "Density field 'density' not found in VTU. "
            f"Available point fields: {list(mesh.point_data.keys())}; "
            f"cell fields: {list(mesh.cell_data.keys())}"
        )

    # Infer nx, ny from unique coordinates on edges
    n_points = points.shape[0]
    x_vals = points[:, 0]
    y_vals = points[:, 1]

    x_min, x_max = x_vals.min(), x_vals.max()
    y_min, y_max = y_vals.min(), y_vals.max()

    # Count unique y-values along left edge (x=xmin) -> ny+1
    tol = 1e-10
    y_at_xmin = y_vals[np.isclose(x_vals, x_min, atol=tol)]
    ny = len(np.unique(np.round(y_at_xmin, decimals=10))) - 1

    # Count unique x-values along bottom edge (y=ymin) -> nx+1
    x_at_ymin = x_vals[np.isclose(y_vals, y_min, atol=tol)]
    nx = len(np.unique(np.round(x_at_ymin, decimals=10))) - 1

    # Verify
    expected_points = (nx + 1) * (ny + 1)
    expected_cells = nx * ny
    if expected_points != n_points or expected_cells != n_cells:
        raise ValueError(
            f"Could not infer grid dimensions. "
            f"Points: {n_points}, cells: {n_cells}, "
            f"inferred nx={nx}, ny={ny} -> expected {expected_points} points, {expected_cells} cells"
        )

    return points, cells, density, nx, ny


# =============================================================================
# feax problem setup
# =============================================================================

def create_compliance_evaluator(
    mesh: Mesh,
    nx: int,
    ny: int,
    bc_name: str = "cantilever_corner",
    *,
    e0: float = E0,
    emin: float = Emin,
    nu: float = NU,
    penal: float = PENAL,
    traction: float = TRACTION,
    fwd_linear_solver: str = "spsolve",
    bwd_linear_solver: str = "spsolve",
    check_convergence: bool = True,
    verbose: bool = False,
):
    """
    Create a compliance evaluator function for a given mesh and BCs.

    Parameters
    ----------
    mesh : Mesh
        feax Mesh with QUAD4 elements.
    nx, ny : int
        Number of elements in x and y directions.
    bc_name : str
        Neumann boundary condition preset (e.g. 'cantilever_corner').
    e0, emin, nu, penal, traction : float
        Material properties.
    fwd_linear_solver, bwd_linear_solver : str
        Linear solver names.

    Returns
    -------
    solve_forward : Callable
        Function ``solve_forward(rho_cells, surface_vars=None) -> compliance``
    evaluate_volume : Callable
        Function ``evaluate_volume(rho_cells) -> volume_fraction``
    problem : PlaneStressElasticityProblem
        The feax problem instance.
    surface_vars : tuple
        Surface variables for the specified boundary condition.
    """
    bc_name = canonicalize_neumann_boundary_conditions(bc_name)

    # Domain dimensions from mesh bbox
    pts = np.array(mesh.points)
    Lx = float(pts[:, 0].max() - pts[:, 0].min())
    Ly = float(pts[:, 1].max() - pts[:, 1].min())

    # Dirichlet BC: left edge clamped
    dirichlet_boundary_conditions = make_dirichlet_boundary_conditions(
        "cantilever_left_support", Lx, Ly
    )

    # Neumann BC location: right edge
    neumann_boundary_location = make_neumann_boundary_location(Lx, Ly)

    # Neumann surface variable function
    surface_var_fn = make_neumann_surface_var_fn(
        bc_name, Lx, Ly, traction_value=traction, Ny=ny
    )

    # Solver setup
    fwd_solver_options, bwd_solver_options, matrix_view = build_solver_setup(
        fwd_linear_solver,
        bwd_linear_solver,
        check_convergence=check_convergence,
        verbose=verbose,
    )

    # Problem definition
    problem = PlaneStressElasticityProblem(
        mesh=mesh,
        vec=2,
        dim=2,
        ele_type="QUAD4",
        gauss_order=2,
        location_fns=[neumann_boundary_location],
        additional_info=(e0, emin, nu, penal, traction),
        matrix_view=matrix_view,
    )

    # Boundary conditions
    bc_config = DirichletBCConfig(list(dirichlet_boundary_conditions))
    bc = bc_config.create_bc(problem)

    # Initial guess
    initial_guess = zero_like_initial_guess(problem, bc)

    # Dynamic compliance function
    dynamic_compliance_fn = create_dynamic_compliance_fn(problem)

    # Solver
    zero_surface_vars = (
        (InternalVars.create_uniform_surface_var(problem, 0.0),),
    )
    sample_density = 0.5
    sample_internal_vars = InternalVars(
        volume_vars=(InternalVars.create_cell_var(problem, sample_density),),
        surface_vars=zero_surface_vars,
    )

    solver = create_solver(
        problem,
        bc=bc,
        solver_options=fwd_solver_options,
        adjoint_solver_options=bwd_solver_options,
        iter_num=1,
        internal_vars=sample_internal_vars,
    )

    def solve_forward(rho_cells, surface_vars=None):
        """Compute compliance for given cell-based density field."""
        if surface_vars is None:
            surface_vars = zero_surface_vars
        internal_vars = InternalVars(
            volume_vars=(rho_cells,),
            surface_vars=surface_vars,
        )
        sol = solver(internal_vars, initial_guess)
        return dynamic_compliance_fn(sol, surface_vars)

    # Volume function
    volume_fn = create_volume_fn(problem)

    def evaluate_volume(rho_cells):
        return volume_fn(rho_cells)

    # Create surface vars for this BC
    surface_vars = create_surface_vars(problem, (surface_var_fn,))[0]

    return solve_forward, evaluate_volume, problem, surface_vars


def _equivalent_traction_for_point_load(
    bc_name: str,
    *,
    point_load_magnitude: float,
    Ly: float,
    ny: int,
) -> float:
    """Map a target total point load to a traction value on the chosen BC patch."""
    canonical_bc = canonicalize_neumann_boundary_conditions(bc_name)
    if ny <= 0 or Ly <= 0.0:
        raise ValueError(f"Invalid mesh dimensions for load matching: Ly={Ly}, ny={ny}")

    edge_segment_length = Ly / ny
    loaded_segment_count = {
        "cantilever_corner": 1,
        "cantilever_mid": 1,
        "cantilever_top_corner": 1,
        "cantilever_upper_mid": 1,
        "cantilever_lower_mid": 1,
        "cantilever_double": 2,
        "cantilever_triple": 3,
        "cantilever_endcaps": 2,
        "cantilever_full": ny,
    }[canonical_bc]
    loaded_length = loaded_segment_count * edge_segment_length
    return point_load_magnitude / loaded_length


# =============================================================================
# Main evaluation
# =============================================================================

def evaluate_vtu(
    vtu_path: str | Path,
    bc_name: str = "cantilever_corner",
    *,
    e0: float = E0,
    emin: float = Emin,
    nu: float = NU,
    penal: float = PENAL,
    traction: float = TRACTION,
    match_point_load: bool = False,
    point_load_magnitude: float = POINT_LOAD_MAGNITUDE,
    fwd_linear_solver: str = "spsolve",
    verbose: bool = False,
):
    """
    Evaluate compliance of a TOuNN .vtu file.

    Parameters
    ----------
    vtu_path : str | Path
        Path to the .vtu file.
    bc_name : str
        Neumann boundary condition preset.
    e0, emin, nu, penal, traction : float
        Material properties (default values are hardcoded).
    fwd_linear_solver : str
        Linear solver for forward solve.
    verbose : bool
        Print verbose output.

    Returns
    -------
    dict
        Dictionary with compliance, volume_fraction, and other metrics.
    """
    vtu_path = Path(vtu_path)
    print(f"Reading VTU: {vtu_path}")

    # Parse VTU
    points, cells, density, nx, ny = parse_vtu_raw(vtu_path)
    n_cells = cells.shape[0]
    print(f"  Mesh: {points.shape[0]} nodes, {n_cells} cells ({nx} x {ny})")
    print(f"  Domain: x=[{points[:, 0].min():.4f}, {points[:, 0].max():.4f}], "
          f"y=[{points[:, 1].min():.4f}, {points[:, 1].max():.4f}]")
    print(f"  Density: min={density.min():.6f}, mean={density.mean():.6f}, "
          f"max={density.max():.6f}")
    Ly = float(points[:, 1].max() - points[:, 1].min())
    if match_point_load:
        traction = _equivalent_traction_for_point_load(
            bc_name,
            point_load_magnitude=point_load_magnitude,
            Ly=Ly,
            ny=ny,
        )
        print(
            "  Load matching: enabled "
            f"(point_load={point_load_magnitude}, traction={traction:.6g})"
        )

    # Build feax Mesh from VTU data
    feax_mesh = Mesh(points, cells, ele_type="QUAD4")

    # Create compliance evaluator
    print(f"\nSetting up feax solver...")
    print(f"  Material: E0={e0}, Emin={emin}, nu={nu}, penal={penal}")
    print(f"  Traction: {traction}")
    print(f"  BC: {bc_name}")

    solve_forward, evaluate_volume, problem, surface_vars = create_compliance_evaluator(
        feax_mesh,
        nx=nx,
        ny=ny,
        bc_name=bc_name,
        e0=e0,
        emin=emin,
        nu=nu,
        penal=penal,
        traction=traction,
        fwd_linear_solver=fwd_linear_solver,
        verbose=verbose,
    )

    # Convert density to JAX array
    rho_cells = jnp.array(density)

    # Evaluate compliance
    print("\nEvaluating compliance...")
    compliance = float(solve_forward(rho_cells, surface_vars))
    volume_fraction = float(evaluate_volume(rho_cells))

    print(f"\n{'='*50}")
    print(f"Results for: {vtu_path.name}")
    print(f"{'='*50}")
    print(f"  Compliance:        {compliance:.6f}")
    print(f"  Volume fraction:   {volume_fraction:.6f}")
    print(f"  Density min/mean/max: {density.min():.4f} / {density.mean():.4f} / {density.max():.4f}")

    return {
        "compliance": compliance,
        "volume_fraction": volume_fraction,
        "density_min": float(density.min()),
        "density_max": float(density.max()),
        "density_mean": float(density.mean()),
        "num_cells": n_cells,
        "num_nodes": points.shape[0],
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate compliance of a TOuNN .vtu file using feax."
    )
    parser.add_argument("vtu_path", type=Path, help="Path to the .vtu file")
    parser.add_argument(
        "--bc",
        type=str,
        default="cantilever_corner",
        help="Neumann boundary condition preset (default: cantilever_corner)",
    )
    parser.add_argument(
        "--e0",
        type=float,
        default=E0,
        help=f"Young's modulus of solid material (default: {E0})",
    )
    parser.add_argument(
        "--emin",
        type=float,
        default=Emin,
        help=f"Young's modulus of void material (default: {Emin})",
    )
    parser.add_argument(
        "--nu",
        type=float,
        default=NU,
        help=f"Poisson's ratio (default: {NU})",
    )
    parser.add_argument(
        "--penal",
        type=float,
        default=PENAL,
        help=f"SIMP penalization exponent (default: {PENAL})",
    )
    parser.add_argument(
        "--traction",
        type=float,
        default=TRACTION,
        help=f"Traction magnitude (default: {TRACTION})",
    )
    parser.add_argument(
        "--match-point-load",
        action="store_true",
        help=(
            "Set traction so total applied boundary force matches a target "
            "point load magnitude (for TOuNN comparability)."
        ),
    )
    parser.add_argument(
        "--point-load",
        type=float,
        default=POINT_LOAD_MAGNITUDE,
        help=(
            "Target point-load magnitude used with --match-point-load "
            f"(default: {POINT_LOAD_MAGNITUDE})."
        ),
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="spsolve",
        help="Linear solver (default: spsolve)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose solver output",
    )

    args = parser.parse_args()

    evaluate_vtu(
        args.vtu_path,
        bc_name=args.bc,
        e0=args.e0,
        emin=args.emin,
        nu=args.nu,
        penal=args.penal,
        traction=args.traction,
        match_point_load=args.match_point_load,
        point_load_magnitude=args.point_load,
        fwd_linear_solver=args.solver,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
