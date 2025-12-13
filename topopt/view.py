#!/usr/bin/env python3
"""
view_models.py — visualize density predictions from multiple models
saved via `serialize_ensemble`.

Usage:
    python view_models.py --dir outputs/2025-11-02_13-51-04 --scale 5 --domain 60,30

Keyboard:
    left/right arrow keys to switch pages if there are many models.
"""

import argparse
from pathlib import Path

import jax.numpy as np
import matplotlib.pyplot as plt
from feax.mesh import rectangle_mesh
from fem_utils import get_element_geometry
from jax.nn import sigmoid
from serialization import load_model_from_config
from visualize import show_rho_pages


def predict_density(model, Lx, Ly, Nx, Ny):
    mesh = rectangle_mesh(Nx=Nx, Ny=Ny, domain_x=Lx, domain_y=Ly)
    coords = get_element_geometry(mesh)["centroids_scaled"]
    rho_pred = sigmoid(model(coords))
    rho_pred = np.reshape(rho_pred, (Ny, Nx), order="F")
    return rho_pred


def main():
    parser = argparse.ArgumentParser(
        description="Visualize model outputs saved by serialize_ensemble."
    )
    parser.add_argument(
        "--dir", type=str, help="Directory containing model files + *_config.json."
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=5,
        help="Scaling factor for resolution (default: 5).",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="60,30",
        help="Domain size as Lx,Ly (default: 60,30).",
    )
    args = parser.parse_args()

    if args.dir:
        base_dir = (Path.cwd() / args.dir).resolve()
    else:
        outputs_dir = (Path.cwd() / "outputs").resolve()
        if not outputs_dir.exists() or not any(outputs_dir.iterdir()):
            print("No output directories found in ./outputs")
            return
        base_dir = sorted(outputs_dir.iterdir())[-1]

    if not base_dir.exists():
        print(f"Directory not found: {base_dir}")
        return

    print(f"Using directory: {base_dir}")

    Lx, Ly = map(float, args.domain.split(","))
    scale = args.scale
    Nx, Ny = int(Lx * scale), int(Ly * scale)

    config_files = sorted(base_dir.glob("*_config.json"))
    if not config_files:
        print(f"No per-model config files (*_config.json) found in {base_dir}")
        return

    print(f"Found {len(config_files)} model configs in {base_dir}")
    images, titles = [], []

    for cfg_path in config_files:
        print(f"Loading from {cfg_path.name} ...")
        model, target_density, penalty, cfg = load_model_from_config(cfg_path, base_dir)
        rho_pred = predict_density(model, Lx, Ly, Nx, Ny)
        images.append(np.array(rho_pred))

        actual_density = float(np.mean(rho_pred))

        base_name = cfg_path.stem.replace("_config", "")
        title = base_name
        if target_density is not None:
            title += f"\nρ*={target_density:.2f}"
        title += f"\nρ_actual={actual_density:.3f}"
        if penalty is not None:
            title += f"\npenalty={penalty}"
        if cfg["model_kwargs"].get("omega") is not None:
            title += f"\nomega={cfg['model_kwargs']['omega']}"

        titles.append(title)

        print(
            f"  Range: [{rho_pred.min():.3f}, {rho_pred.max():.3f}], "
            f"mean ρ = {actual_density:.3f}"
        )

        out_path = base_dir / f"{base_name}_rho.png"
        plt.imsave(out_path, rho_pred, cmap="gray_r", origin="lower")

    show_rho_pages(images, titles, Nx=Nx, Ny=Ny, per_page=6)


if __name__ == "__main__":
    main()
