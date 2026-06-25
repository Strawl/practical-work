from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from topopt.upscale_vtu import load_density_from_vtu


def point_grid_to_cell_grid(rho_grid: np.ndarray) -> np.ndarray:
    return 0.25 * (
        rho_grid[:-1, :-1]
        + rho_grid[1:, :-1]
        + rho_grid[:-1, 1:]
        + rho_grid[1:, 1:]
    )


def save_plain_density_png(rho_grid: np.ndarray, path: Path) -> None:
    rho_img = np.flipud(rho_grid.T)
    rho_img = np.clip(rho_img, 0.0, 1.0)
    grayscale = np.rint((1.0 - rho_img) * 255.0).astype(np.uint8)
    image = Image.fromarray(grayscale, mode="L").resize((3000, 1500), Image.Resampling.NEAREST)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def export_plain_image(vtu_path: Path, output_path: Path) -> None:
    rho_grid, source_is_point_data, *_ = load_density_from_vtu(vtu_path)
    if source_is_point_data:
        rho_grid = point_grid_to_cell_grid(rho_grid)
    save_plain_density_png(rho_grid, output_path)
    print(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate plain baseline density PNGs without titles."
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path("outputs/06-23_17-56-03_baseline_50_iterations"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/12"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_plain_image(
        args.baseline_dir / "final.vtu",
        args.output_dir / "mma_50_scale3_projected.png",
    )
    export_plain_image(
        args.baseline_dir / "baseline_final_with_without_filter_upscaled_x5.vtu",
        args.output_dir / "mma_50_scale15_projected.png",
    )


if __name__ == "__main__":
    main()
