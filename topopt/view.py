#!/usr/bin/env python3
"""
view_models.py — visualize density predictions from multiple models
saved via `serialize_ensemble`.

Usage:
    python view_models.py --dir outputs/2025-11-02_13-51-04 --scale 5 --domain 60,30

Keyboard:
    left/right arrow keys to switch pages if there are many models.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import config
import jax.numpy as jnp
import matplotlib.pyplot as plt
from feax.mesh import rectangle_mesh
from fem_utils import get_element_geometry
from jax.nn import sigmoid
from serialization import (
    TrainingConfig,
    load_model_from_config,
)
from visualize import show_rho_pages


def predict_density(model, Lx: float, Ly: float, Nx: int, Ny: int):
    mesh = rectangle_mesh(Nx=Nx, Ny=Ny, domain_x=Lx, domain_y=Ly)
    coords = get_element_geometry(mesh)["centroids_scaled"]
    rho_pred = sigmoid(model(coords))
    rho_pred = jnp.reshape(rho_pred, (Ny, Nx), order="F")
    return rho_pred


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize model outputs saved by serialize_ensemble."
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=None,
        help="Override resolution scaling factor (defaults to training_config_snapshot.yaml).",
    )
    args = parser.parse_args()

    base_dir = Path(config.SAVE_DIR).resolve()
    if not base_dir.exists():
        print(f"SAVE_DIR not found: {base_dir}")
        return

    print(f"Using directory (SAVE_DIR): {base_dir}")

    train_config_path = base_dir / "training_config_snapshot.yaml"
    train_config: TrainingConfig = TrainingConfig.from_yaml(train_config_path)

    Lx = float(train_config.training.Lx)
    Ly = float(train_config.training.Ly)
    scale = (
        int(args.scale) if args.scale is not None else int(train_config.training.scale)
    )

    Nx, Ny = int(Lx * scale), int(Ly * scale)
    print(f"Domain: Lx={Lx}, Ly={Ly} | scale={scale} -> Nx={Nx}, Ny={Ny}")

    config_files = sorted(base_dir.glob("*_config.yaml"))
    if not config_files:
        print(f"No per-model config files (*_config.yaml) found in {base_dir}")
        return

    print(f"Found {len(config_files)} model configs in {base_dir}")
    images, titles = [], []

    for cfg_path in config_files:
        print(f"Loading from {cfg_path.name} ...")

        model, _, _, cfg = load_model_from_config(cfg_path, base_dir)

        rho_pred = predict_density(model, Lx, Ly, Nx, Ny)
        rho_pred = jnp.array(rho_pred)
        images.append(rho_pred)

        actual_density = float(jnp.mean(rho_pred))
        weights_file = cfg.get("weights_file")
        base_name = Path(weights_file).stem

        title = base_name
        training = cfg.get("training", {}) if isinstance(cfg, dict) else {}
        td = training.get("target_density")
        pen = training.get("penalty")

        if td is not None:
            title += f"\nρ*={float(td):.2f}"
        title += f"\nρ_actual={actual_density:.3f}"
        if pen is not None:
            title += f"\npenalty={pen:g}"

        model_kwargs = cfg.get("model_kwargs", {}) if isinstance(cfg, dict) else {}
        if isinstance(model_kwargs, dict) and model_kwargs.get("omega") is not None:
            title += f"\nomega={model_kwargs['omega']}"

        titles.append(title)

        print(
            f"  Range: [{float(rho_pred.min()):.3f}, {float(rho_pred.max()):.3f}], "
            f"mean ρ = {actual_density:.3f}"
        )

        out_path = base_dir / f"{base_name}_rho.png"
        plt.imsave(out_path, rho_pred, cmap="gray_r", origin="lower")

    show_rho_pages(images, titles, Nx=Nx, Ny=Ny, per_page=6)


if __name__ == "__main__":
    main()
