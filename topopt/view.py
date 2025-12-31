from __future__ import annotations

from pathlib import Path
from typing import Optional

import jax.numpy as jnp
from feax.mesh import rectangle_mesh
from topopt.fem_utils import get_element_geometry
from jax.nn import sigmoid

from topopt.serialization import (
    TrainingConfig,
    load_model_from_config,
)
from topopt.visualize import save_rho_png, show_rho_pages


def predict_density(model, Lx: float, Ly: float, Nx: int, Ny: int):
    mesh = rectangle_mesh(Nx=Nx, Ny=Ny, domain_x=Lx, domain_y=Ly)
    coords = get_element_geometry(mesh)["centroids_scaled"]
    rho_pred = sigmoid(model(coords))
    rho_pred = jnp.reshape(rho_pred, (Ny, Nx), order="F")
    return rho_pred


def view_outputs(*, save_dir: Path, scale: Optional[int] = None) -> None:
    """
    Visualize model outputs saved by serialize_ensemble.

    Args:
        scale: optional override for resolution scaling factor
        save_dir: optional override directory (normally comes from config.SAVE_DIR)
    """
    if not save_dir.exists():
        print(f"SAVE_DIR not found: {save_dir}")
        return

    print(f"Using directory (SAVE_DIR): {save_dir}")

    train_config_path = save_dir / "training_config_snapshot.yaml"
    train_config: TrainingConfig = TrainingConfig.from_yaml(train_config_path)

    Lx = float(train_config.training.Lx)
    Ly = float(train_config.training.Ly)
    used_scale = int(scale) if scale is not None else int(train_config.training.scale)

    Nx, Ny = int(Lx * used_scale), int(Ly * used_scale)
    print(f"Domain: Lx={Lx}, Ly={Ly} | scale={used_scale} -> Nx={Nx}, Ny={Ny}")

    config_files = sorted(save_dir.glob("*_config.yaml"))
    if not config_files:
        print(f"No per-model config files (*_config.yaml) found in {save_dir}")
        return

    print(f"Found {len(config_files)} model configs in {save_dir}")
    images, titles = [], []

    for cfg_path in config_files:
        print(f"Loading from {cfg_path.name} ...")

        model, _, _, cfg = load_model_from_config(cfg_path, save_dir)

        rho_pred = predict_density(model, Lx, Ly, Nx, Ny)
        rho_pred = jnp.array(rho_pred)
        images.append(rho_pred)

        actual_density = float(jnp.mean(rho_pred))
        weights_file = cfg.get("weights_file")
        base_name = Path(weights_file).stem if weights_file else cfg_path.stem

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

        out_path = save_dir / f"{base_name}_rho.png"
        save_rho_png(rho_pred, title, Nx, Ny, out_path)

    show_rho_pages(images, titles, Nx=Nx, Ny=Ny, per_page=6)
