from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from feax.mesh import rectangle_mesh

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Upscale a baseline density field by interpolating the coarse field, "
            "applying density-filter smoothing, and then a Heaviside projection."
        )
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Input baseline .npz or VTU with point_data['density'] or cell_data['density'].",
    )
    parser.add_argument(
        "--scale-factor",
        type=int,
        default=3,
        help="Resolution multiplier for the output mesh.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=None,
        help=(
            "Density-filter radius in physical units. "
            "Defaults to one coarse-cell width."
        ),
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=30.0,
        help="Heaviside sharpness parameter.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Heaviside threshold parameter.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for exported PNG and VTU files. Defaults next to the input VTU.",
    )
    parser.add_argument(
        "--array-name",
        type=str,
        default="rho_filtered",
        help=(
            "Array to read from baseline .npz. "
            "Typical choices: rho_unfiltered, rho_filtered, rho_projected."
        ),
    )
    parser.add_argument(
        "--reference-vtu",
        type=Path,
        default=None,
        help=(
            "Structured VTU used to infer mesh/domain metadata when the input is .npz. "
            "Defaults to final.vtu next to the .npz file if present."
        ),
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Output filename prefix. Defaults to '<input_stem>_upscaled_x<scale-factor>'.",
    )
    return parser.parse_args()


def infer_grid_metadata(points: np.ndarray) -> tuple[int, int, float, float, float, float]:
    x0 = float(points[:, 0].min())
    x1 = float(points[:, 0].max())
    y0 = float(points[:, 1].min())
    y1 = float(points[:, 1].max())

    tol = 1e-10
    y_at_xmin = points[:, 1][np.isclose(points[:, 0], x0, atol=tol)]
    x_at_ymin = points[:, 0][np.isclose(points[:, 1], y0, atol=tol)]
    ny = len(np.unique(np.round(y_at_xmin, decimals=10))) - 1
    nx = len(np.unique(np.round(x_at_ymin, decimals=10))) - 1

    expected_points = (nx + 1) * (ny + 1)
    if expected_points != points.shape[0]:
        raise ValueError(
            "Could not infer structured grid dimensions from VTU points. "
            f"Found {points.shape[0]} points but inferred nx={nx}, ny={ny} "
            f"which implies {expected_points} structured nodes."
        )

    return nx, ny, x0, x1, y0, y1


def density_to_cell_grid(
    points: np.ndarray,
    cells: np.ndarray,
    density: np.ndarray,
    nx: int,
    ny: int,
) -> np.ndarray:
    x0 = float(points[:, 0].min())
    x1 = float(points[:, 0].max())
    y0 = float(points[:, 1].min())
    y1 = float(points[:, 1].max())
    dx = (x1 - x0) / nx
    dy = (y1 - y0) / ny

    centroids = points[cells].mean(axis=1)
    rho_grid = np.zeros((nx, ny), dtype=np.float64)

    ix = np.clip(np.floor((centroids[:, 0] - x0) / dx).astype(int), 0, nx - 1)
    iy = np.clip(np.floor((centroids[:, 1] - y0) / dy).astype(int), 0, ny - 1)
    rho_grid[ix, iy] = density

    return rho_grid


def density_to_point_grid(
    points: np.ndarray,
    density: np.ndarray,
    nx: int,
    ny: int,
) -> np.ndarray:
    x0 = float(points[:, 0].min())
    x1 = float(points[:, 0].max())
    y0 = float(points[:, 1].min())
    y1 = float(points[:, 1].max())
    dx = (x1 - x0) / nx
    dy = (y1 - y0) / ny

    rho_grid = np.zeros((nx + 1, ny + 1), dtype=np.float64)
    ix = np.clip(np.rint((points[:, 0] - x0) / dx).astype(int), 0, nx)
    iy = np.clip(np.rint((points[:, 1] - y0) / dy).astype(int), 0, ny)
    rho_grid[ix, iy] = density
    return rho_grid


def interpolate_grid(
    rho_grid: np.ndarray,
    *,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    fine_nx: int,
    fine_ny: int,
    cell_centered: bool,
) -> tuple[np.ndarray, np.ndarray]:
    coarse_nx, coarse_ny = rho_grid.shape

    if cell_centered:
        dx_coarse = (x1 - x0) / coarse_nx
        dy_coarse = (y1 - y0) / coarse_ny
        dx_fine = (x1 - x0) / fine_nx
        dy_fine = (y1 - y0) / fine_ny
        coarse_x = x0 + (np.arange(coarse_nx) + 0.5) * dx_coarse
        coarse_y = y0 + (np.arange(coarse_ny) + 0.5) * dy_coarse
        fine_x = x0 + (np.arange(fine_nx) + 0.5) * dx_fine
        fine_y = y0 + (np.arange(fine_ny) + 0.5) * dy_fine
    else:
        dx_coarse = (x1 - x0) / (coarse_nx - 1)
        dy_coarse = (y1 - y0) / (coarse_ny - 1)
        dx_fine = (x1 - x0) / fine_nx
        dy_fine = (y1 - y0) / fine_ny
        coarse_x = x0 + np.arange(coarse_nx) * dx_coarse
        coarse_y = y0 + np.arange(coarse_ny) * dy_coarse
        fine_x = x0 + np.arange(fine_nx + 1) * dx_fine
        fine_y = y0 + np.arange(fine_ny + 1) * dy_fine

    interp_y = np.empty((coarse_nx, fine_y.shape[0]), dtype=np.float64)
    for ix in range(coarse_nx):
        interp_y[ix, :] = np.interp(fine_y, coarse_y, rho_grid[ix, :])

    interp_xy = np.empty((fine_x.shape[0], fine_y.shape[0]), dtype=np.float64)
    for iy in range(fine_y.shape[0]):
        interp_xy[:, iy] = np.interp(fine_x, coarse_x, interp_y[:, iy])

    fine_coords_x, fine_coords_y = np.meshgrid(fine_x, fine_y, indexing="ij")
    fine_coords = np.stack(
        [fine_coords_x.reshape(-1, order="C"), fine_coords_y.reshape(-1, order="C")],
        axis=1,
    )
    return interp_xy, fine_coords


def structured_cone_filter(rho_grid: np.ndarray, radius_x: float, radius_y: float) -> np.ndarray:
    if radius_x <= 0.0 or radius_y <= 0.0:
        return rho_grid.copy()

    rx = int(np.ceil(radius_x))
    ry = int(np.ceil(radius_y))
    filtered = np.zeros_like(rho_grid, dtype=np.float64)
    weights_sum = np.zeros_like(rho_grid, dtype=np.float64)

    nx, ny = rho_grid.shape
    for dx in range(-rx, rx + 1):
        for dy in range(-ry, ry + 1):
            dist = np.sqrt((dx / radius_x) ** 2 + (dy / radius_y) ** 2)
            if dist > 1.0:
                continue

            weight = 1.0 - dist
            x_src_start = max(0, -dx)
            x_src_end = min(nx, nx - dx)
            y_src_start = max(0, -dy)
            y_src_end = min(ny, ny - dy)
            x_dst_start = max(0, dx)
            x_dst_end = min(nx, nx + dx)
            y_dst_start = max(0, dy)
            y_dst_end = min(ny, ny + dy)

            filtered[x_dst_start:x_dst_end, y_dst_start:y_dst_end] += (
                weight * rho_grid[x_src_start:x_src_end, y_src_start:y_src_end]
            )
            weights_sum[x_dst_start:x_dst_end, y_dst_start:y_dst_end] += weight

    return filtered / np.maximum(weights_sum, 1e-12)


def heaviside_projection_numpy(
    rho_grid: np.ndarray,
    *,
    beta: float,
    threshold: float,
) -> np.ndarray:
    return (np.tanh(beta * (rho_grid - threshold)) + 1.0) / 2.0


def save_density_png(
    rho_grid: np.ndarray,
    *,
    path: Path,
    cmap: str = "gray_r",
    dpi: int = 300,
) -> None:
    image = rho_grid.T
    fig = plt.figure(figsize=(5, 5), frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(image, cmap=cmap, origin="lower", vmin=0.0, vmax=1.0)
    ax.set_axis_off()
    fig.savefig(path, dpi=dpi, pad_inches=0)
    plt.close(fig)


def export_cell_density_vtu(
    rho_interpolated: np.ndarray,
    rho_filtered: np.ndarray,
    rho_projected: np.ndarray,
    *,
    fine_nx: int,
    fine_ny: int,
    x0: float,
    y0: float,
    lx: float,
    ly: float,
    output_path: Path,
) -> None:
    mesh = rectangle_mesh(
        Nx=fine_nx,
        Ny=fine_ny,
        domain_x=lx,
        domain_y=ly,
        origin=(x0, y0),
        ele_type="QUAD4",
    )
    points = np.column_stack([np.asarray(mesh.points), np.zeros(mesh.points.shape[0])])
    cells = np.hstack(
        [
            np.full((mesh.cells.shape[0], 1), 4, dtype=np.int64),
            np.asarray(mesh.cells, dtype=np.int64),
        ]
    ).ravel()
    cell_types = np.full(mesh.cells.shape[0], pv.CellType.QUAD, dtype=np.uint8)

    grid = pv.UnstructuredGrid(cells, cell_types, points)
    grid.cell_data["density"] = rho_projected.reshape(-1, order="C")
    grid.cell_data["density_interpolated"] = rho_interpolated.reshape(-1, order="C")
    grid.cell_data["density_filtered"] = rho_filtered.reshape(-1, order="C")
    grid.cell_data["density_projected"] = rho_projected.reshape(-1, order="C")
    grid.save(output_path)


def export_point_density_vtu(
    rho_interpolated: np.ndarray,
    rho_filtered: np.ndarray,
    rho_projected: np.ndarray,
    *,
    fine_nx: int,
    fine_ny: int,
    x0: float,
    y0: float,
    lx: float,
    ly: float,
    output_path: Path,
) -> None:
    mesh = rectangle_mesh(
        Nx=fine_nx,
        Ny=fine_ny,
        domain_x=lx,
        domain_y=ly,
        origin=(x0, y0),
        ele_type="QUAD4",
    )
    points = np.column_stack([np.asarray(mesh.points), np.zeros(mesh.points.shape[0])])
    cells = np.hstack(
        [
            np.full((mesh.cells.shape[0], 1), 4, dtype=np.int64),
            np.asarray(mesh.cells, dtype=np.int64),
        ]
    ).ravel()
    cell_types = np.full(mesh.cells.shape[0], pv.CellType.QUAD, dtype=np.uint8)

    grid = pv.UnstructuredGrid(cells, cell_types, points)
    grid.point_data["density"] = rho_projected.reshape(-1, order="C")
    grid.point_data["density_interpolated"] = rho_interpolated.reshape(-1, order="C")
    grid.point_data["density_filtered"] = rho_filtered.reshape(-1, order="C")
    grid.point_data["density_projected"] = rho_projected.reshape(-1, order="C")
    grid.save(output_path)


def load_density_from_vtu(vtu_path: Path) -> tuple[np.ndarray, bool, int, int, float, float, float, float]:
    mesh = pv.read(vtu_path)
    points_3d = np.asarray(mesh.points, dtype=np.float64)
    if not np.allclose(points_3d[:, 2], 0.0):
        raise ValueError("VTU mesh is not planar (z != 0). Expected 2D mesh.")

    points = points_3d[:, :2]
    nx, ny, x0, x1, y0, y1 = infer_grid_metadata(points)

    if "density" in mesh.point_data:
        return (
            density_to_point_grid(
                points,
                np.asarray(mesh.point_data["density"], dtype=np.float64).ravel(),
                nx,
                ny,
            ),
            True,
            nx,
            ny,
            x0,
            x1,
            y0,
            y1,
        )

    if "density" in mesh.cell_data:
        cells_flat = np.asarray(mesh.cells)
        n_cells = mesh.n_cells
        cells = cells_flat.reshape(n_cells, 5)[:, 1:].astype(np.int32)
        return (
            density_to_cell_grid(
                points,
                cells,
                np.asarray(mesh.cell_data["density"], dtype=np.float64).ravel(),
                nx,
                ny,
            ),
            False,
            nx,
            ny,
            x0,
            x1,
            y0,
            y1,
        )

    raise ValueError(
        "Density field not found in VTU. "
        f"Available point fields: {list(mesh.point_data.keys())}; "
        f"cell fields: {list(mesh.cell_data.keys())}"
    )


def resolve_reference_vtu(npz_path: Path, reference_vtu: Path | None) -> Path:
    if reference_vtu is not None:
        return reference_vtu

    default_vtu = npz_path.with_name("final.vtu")
    if default_vtu.exists():
        return default_vtu

    raise FileNotFoundError(
        "Input is .npz but no reference VTU was provided and "
        f"{default_vtu} does not exist."
    )


def load_density_from_npz(
    npz_path: Path,
    array_name: str,
    reference_vtu: Path | None,
) -> tuple[np.ndarray, bool, int, int, float, float, float, float]:
    data = np.load(npz_path)
    if array_name not in data:
        raise KeyError(
            f"Array '{array_name}' not found in {npz_path}. "
            f"Available arrays: {sorted(data.files)}"
        )

    ref_vtu = resolve_reference_vtu(npz_path, reference_vtu)
    _, source_is_point_data, nx, ny, x0, x1, y0, y1 = load_density_from_vtu(ref_vtu)

    rho = np.asarray(data[array_name], dtype=np.float64).ravel()
    expected_point_count = (nx + 1) * (ny + 1)
    expected_cell_count = nx * ny

    if rho.size == expected_point_count:
        rho_grid = rho.reshape((nx + 1, ny + 1), order="C")
        return rho_grid, True, nx, ny, x0, x1, y0, y1
    if rho.size == expected_cell_count:
        rho_grid = rho.reshape((nx, ny), order="C")
        return rho_grid, False, nx, ny, x0, x1, y0, y1

    if source_is_point_data:
        raise ValueError(
            f"Array '{array_name}' has length {rho.size}, expected {expected_point_count} "
            f"for point-based data inferred from {ref_vtu}."
        )

    raise ValueError(
        f"Array '{array_name}' has length {rho.size}, expected {expected_cell_count} "
        f"for cell-based data inferred from {ref_vtu}."
    )


def main() -> None:
    args = parse_args()

    if args.scale_factor < 1:
        raise ValueError("--scale-factor must be >= 1")

    input_path = args.input_path
    if input_path.suffix == ".npz":
        rho_coarse, source_is_point_data, nx, ny, x0, x1, y0, y1 = load_density_from_npz(
            input_path,
            args.array_name,
            args.reference_vtu,
        )
    elif input_path.suffix == ".vtu":
        rho_coarse, source_is_point_data, nx, ny, x0, x1, y0, y1 = load_density_from_vtu(
            input_path
        )
    else:
        raise ValueError(
            f"Unsupported input type '{input_path.suffix}'. Use .npz or .vtu."
        )

    fine_nx = nx * args.scale_factor
    fine_ny = ny * args.scale_factor
    rho_interpolated, fine_coords = interpolate_grid(
        rho_coarse,
        x0=x0,
        x1=x1,
        y0=y0,
        y1=y1,
        fine_nx=fine_nx,
        fine_ny=fine_ny,
        cell_centered=not source_is_point_data,
    )

    lx = x1 - x0
    ly = y1 - y0
    coarse_dx = lx / nx
    coarse_dy = ly / ny
    radius = (
        float(args.radius)
        if args.radius is not None
        else max(coarse_dx, coarse_dy)
    )

    del fine_coords

    fine_dx = lx / fine_nx
    fine_dy = ly / fine_ny
    radius_x = radius / fine_dx
    radius_y = radius / fine_dy

    rho_interpolated = np.clip(rho_interpolated, 0.0, 1.0)
    rho_filtered = np.clip(
        structured_cone_filter(rho_interpolated, radius_x=radius_x, radius_y=radius_y),
        0.0,
        1.0,
    )
    rho_projected = np.clip(
        heaviside_projection_numpy(
            rho_filtered,
            beta=args.beta,
            threshold=args.threshold,
        ),
        0.0,
        1.0,
    )

    output_dir = args.output_dir or input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or f"{input_path.stem}_upscaled_x{args.scale_factor}"

    if source_is_point_data:
        export_point_density_vtu(
            rho_interpolated,
            rho_filtered,
            rho_projected,
            fine_nx=fine_nx,
            fine_ny=fine_ny,
            x0=x0,
            y0=y0,
            lx=lx,
            ly=ly,
            output_path=output_dir / f"{prefix}.vtu",
        )
    else:
        export_cell_density_vtu(
            rho_interpolated,
            rho_filtered,
            rho_projected,
            fine_nx=fine_nx,
            fine_ny=fine_ny,
            x0=x0,
            y0=y0,
            lx=lx,
            ly=ly,
            output_path=output_dir / f"{prefix}.vtu",
        )
    save_density_png(
        rho_interpolated,
        path=output_dir / f"{prefix}_interpolated.png",
    )
    save_density_png(
        rho_filtered,
        path=output_dir / f"{prefix}_filtered.png",
    )
    save_density_png(
        rho_projected,
        path=output_dir / f"{prefix}.png",
    )

    print(f"Input mesh  : {nx} x {ny}")
    print(f"Output mesh : {fine_nx} x {fine_ny}")
    print(f"Density data: {'point_data' if source_is_point_data else 'cell_data'}")
    print(f"Filter radius: {radius:.6f}")
    print(f"Filter radius on fine grid: rx={radius_x:.2f}, ry={radius_y:.2f}")
    print(f"Saved VTU   : {output_dir / f'{prefix}.vtu'}")
    print(f"Saved PNG   : {output_dir / f'{prefix}.png'}")


if __name__ == "__main__":
    main()
