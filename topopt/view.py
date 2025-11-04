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
import json
from pathlib import Path

import jax
import jax.numpy as np
import equinox as eqx
import matplotlib.pyplot as plt

from feax.mesh import rectangle_mesh
from jax.nn import sigmoid
from siren import SIREN
from utils import get_element_centroids


def load_model_from_config(cfg_path: Path, base_dir: Path):
    """Load a trained model using its per-model config JSON.

    Assumes the config was created by `serialize_ensemble`, i.e. it contains:
      - model_type
      - model_kwargs
      - training
      - weights_file
    """
    with cfg_path.open("r") as f:
        cfg = json.load(f)

    model_type = cfg.get("model_type", "SIREN")
    model_kwargs = cfg.get("model_kwargs", {})
    training = cfg.get("training", {})
    weights_file = cfg.get("weights_file", None)

    if weights_file is None:
        base = cfg_path.stem.replace("_config", "")
        weights_file = f"{base}.eqx"

    weights_path = base_dir / weights_file

    if model_type != "SIREN":
        raise NotImplementedError(f"model_type '{model_type}' is not supported yet in view_models.py")

    # Build a dummy SIREN from the kwargs in the JSON.
    # The arrays will be overwritten by tree_deserialise_leaves, but static
    # fields (like omega) must match the original model, so we use model_kwargs.
    rng = jax.random.PRNGKey(0)
    siren_dummy = SIREN(
        rng_key=rng,
        **model_kwargs,
    )

    model = eqx.tree_deserialise_leaves(weights_path, siren_dummy)

    target_density = training.get("target_density")
    penalty = training.get("penalty")

    return model, target_density, penalty, cfg


def predict_density(model, Lx, Ly, Nx, Ny):
    mesh = rectangle_mesh(Nx=Nx, Ny=Ny, domain_x=Lx, domain_y=Ly)
    coords = get_element_centroids(mesh)
    rho_pred = sigmoid(model(coords))
    # rho_pred = (rho_pred >= 0.8).astype(rho_pred.dtype)
    rho_pred = np.reshape(rho_pred, (Ny, Nx), order="F")
    return rho_pred


def show_paged_images(images, titles, per_page=6):
    total = len(images)
    pages = (total + per_page - 1) // per_page
    current_page = 0

    def draw_page(page_idx):
        plt.clf()
        start = page_idx * per_page
        end = min(start + per_page, total)
        ncols = 3
        nrows = int(np.ceil((end - start) / ncols))
        for i, (img, title) in enumerate(zip(images[start:end], titles[start:end])):
            ax = plt.subplot(nrows, ncols, i + 1)
            ax.imshow(img, cmap="gray_r", origin="lower")
            ax.set_title(title, fontsize=9)
            ax.axis("off")
        plt.suptitle(f"Page {page_idx + 1}/{pages}", fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.draw()

    def on_key(event):
        nonlocal current_page
        if event.key == "right":
            current_page = (current_page + 1) % pages
        elif event.key == "left":
            current_page = (current_page - 1) % pages
        draw_page(current_page)

    fig = plt.figure(figsize=(10, 6))
    fig.canvas.mpl_connect("key_press_event", on_key)
    draw_page(0)
    plt.show()


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Visualize model outputs saved by serialize_ensemble.")
    parser.add_argument("--dir", type=str, help="Directory containing model files + *_config.json.")
    parser.add_argument("--scale", type=int, default=5, help="Scaling factor for resolution (default: 5).")
    parser.add_argument("--domain", type=str, default="60,30", help="Domain size as Lx,Ly (default: 60,30).")
    args = parser.parse_args()

    # --- Determine base directory ---
    if args.dir:
        base_dir = (Path.cwd() / args.dir).resolve()
    else:
        outputs_dir = (Path.cwd() / "outputs").resolve()
        if not outputs_dir.exists() or not any(outputs_dir.iterdir()):
            print("No output directories found in ./outputs")
            return
        # Take the latest directory (they are already sorted by timestamp)
        base_dir = sorted(outputs_dir.iterdir())[-1]

    # --- Validate directory ---
    if not base_dir.exists():
        print(f"Directory not found: {base_dir}")
        return

    print(f"Using directory: {base_dir}")

    Lx, Ly = map(float, args.domain.split(","))
    scale = args.scale
    Nx, Ny = int(Lx * scale), int(Ly * scale)

    # Detect per-model config files produced by serialize_ensemble
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

        # --- Compute "actual density" as mean of the predicted density ---
        actual_density = float(np.mean(rho_pred))

        base_name = cfg_path.stem.replace("_config", "")
        title = base_name
        if target_density is not None:
            title += f"\nρ*={target_density:.2f}"
        # Add the actual density to the title
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

        # Save visualization image
        out_path = base_dir / f"{base_name}_rho.png"
        plt.imsave(out_path, rho_pred, cmap="gray_r", origin="lower")

    # Interactive viewer (paged if >6)
    show_paged_images(images, titles, per_page=6)


if __name__ == "__main__":
    main()