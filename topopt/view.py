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
from feax.mesh import rectangle_mesh, Mesh
from jax.nn import sigmoid
from serialization import load_model_from_config
from utils import adaptive_rectangle_mesh, get_element_centroids
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from typing import Optional, Tuple

def plot_mesh(
    mesh: Mesh,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    ax: Optional[plt.Axes] = None,
    linewidth: float = 0.5,
):
    """
    Plot a FEAX Mesh in black & white using matplotlib.

    Parameters
    ----------
    mesh : Mesh
        Mesh instance to plot. Uses mesh.points, mesh.cells, mesh.ele_type.
    x_range : (float, float), optional
        (xmin, xmax) for the x-axis. If None, use mesh bounds.
    y_range : (float, float), optional
        (ymin, ymax) for the y-axis. If None, use mesh bounds.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on. If None, a new figure and axes are created.
    linewidth : float, optional
        Line width for element edges.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the mesh plot.
    """
    # Require at least 2D coordinates
    if mesh.points.shape[1] < 2:
        raise ValueError("Need at least 2D points to plot the mesh.")

    xy = mesh.points[:, :2]

    # Edges for each supported element type (using only corner nodes)
    element_edges = {
        # 2D
        'TRI3': [(0, 1), (1, 2), (2, 0)],
        'TRI6': [(0, 1), (1, 2), (2, 0)],            # ignore midside nodes
        'QUAD4': [(0, 1), (1, 2), (2, 3), (3, 0)],
        'QUAD8': [(0, 1), (1, 2), (2, 3), (3, 0)],   # ignore midside nodes

        # 3D (projected to x-y)
        'TET4':  [(0, 1), (1, 2), (2, 0),
                  (0, 3), (1, 3), (2, 3)],
        'TET10': [(0, 1), (1, 2), (2, 0),
                  (0, 3), (1, 3), (2, 3)],          # ignore midside nodes
        'HEX8':  [(0, 1), (1, 2), (2, 3), (3, 0),
                  (4, 5), (5, 6), (6, 7), (7, 4),
                  (0, 4), (1, 5), (2, 6), (3, 7)],
        'HEX20': [(0, 1), (1, 2), (2, 3), (3, 0),
                  (4, 5), (5, 6), (6, 7), (7, 4),
                  (0, 4), (1, 5), (2, 6), (3, 7)],  # ignore midside nodes
        'HEX27': [(0, 1), (1, 2), (2, 3), (3, 0),
                  (4, 5), (5, 6), (6, 7), (7, 4),
                  (0, 4), (1, 5), (2, 6), (3, 7)],
    }

    ele_type = mesh.ele_type.upper()
    if ele_type not in element_edges:
        raise ValueError(
            f"Unsupported ele_type '{mesh.ele_type}' for plotting. "
            f"Supported types: {list(element_edges.keys())}"
        )

    edges = element_edges[ele_type]

    # Build line segments (each segment is [p0, p1] in 2D)
    segments = []
    for cell in mesh.cells:
        for i_local, j_local in edges:
            i = cell[i_local]
            j = cell[j_local]
            segments.append([xy[i], xy[j]])

    # Prepare axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    # White background
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Add all edges as a LineCollection in black
    lc = LineCollection(segments, colors='k', linewidths=linewidth)
    ax.add_collection(lc)

    # Axis limits
    if x_range is not None:
        ax.set_xlim(*x_range)
    else:
        ax.set_xlim(xy[:, 0].min(), xy[:, 0].max())

    if y_range is not None:
        ax.set_ylim(*y_range)
    else:
        ax.set_ylim(xy[:, 1].min(), xy[:, 1].max())

    ax.set_aspect('equal', adjustable='box')

    # Optional: clean look (no ticks)
    ax.set_xticks([])
    ax.set_yticks([])

    return ax

def predict_density(model, Lx, Ly, Nx, Ny):
    mesh = rectangle_mesh(Nx=Nx, Ny=Ny, domain_x=Lx, domain_y=Ly)
    centroids, coords = get_element_centroids(mesh)
    rho_pred = sigmoid(model(coords))
    # new_mesh = adaptive_rectangle_mesh(
        # initial_size=5.0,
        # coords=centroids,
        # values=rho_pred,
        # domain_x=60.0,
        # domain_y=30.0,
        # origin=(0.0, 0.0),
        # max_depth=5,
        # threshold_low=0.05,
        # threshold_high=0.95,
    # )
    rho_pred = np.reshape(rho_pred, (Ny, Nx), order="F")
    # plot_mesh(new_mesh, (0, Lx), (0, Ly), None, 0.2)
    # plt.show()
    return rho_pred


def show_paged_images(images, titles, per_page=6):
    total = len(images)
    pages = (total + per_page - 1) // per_page
    current_page = 0

    def _get_fontsizes(fig):
        w, h = fig.get_size_inches()
        base = min(w, h)

        title_fs = max(6, int(base * 1.5))
        suptitle_fs = max(8, int(base * 2))

        return title_fs, suptitle_fs

    def draw_page(page_idx):
        fig = plt.gcf()
        title_fs, suptitle_fs = _get_fontsizes(fig)

        plt.clf()
        start = page_idx * per_page
        end = min(start + per_page, total)
        ncols = 3
        nrows = int(np.ceil((end - start) / ncols))

        for i, (img, title) in enumerate(zip(images[start:end], titles[start:end])):
            ax = plt.subplot(nrows, ncols, i + 1)
            ax.imshow(img, cmap="gray_r", origin="lower")
            ax.set_title(title, fontsize=title_fs)
            ax.axis("off")

        plt.suptitle(f"Page {page_idx + 1}/{pages}", fontsize=suptitle_fs)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.draw()

    def on_key(event):
        nonlocal current_page
        if event.key == "right":
            current_page = (current_page + 1) % pages
        elif event.key == "left":
            current_page = (current_page - 1) % pages
        draw_page(current_page)

    def on_resize(event):
        # Just redraw current page with updated font sizes
        draw_page(current_page)

    fig = plt.figure(figsize=(10, 6))
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("resize_event", on_resize)

    draw_page(0)
    plt.show()


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

    show_paged_images(images, titles, per_page=6)


if __name__ == "__main__":
    main()
