from __future__ import annotations

from pathlib import Path
from typing import Optional

import jax.numpy as jnp
import pandas as pd
from feax.mesh import Mesh, rectangle_mesh
from jax.nn import sigmoid
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

from topopt.bc import make_bc_preset
from topopt.fem_utils import create_objective_functions, get_element_geometry
from topopt.monitoring import StepTimer
from topopt.serialization import (
    TrainingConfig,
    load_model_from_config,
)


def plot_mesh(mesh: Mesh, linewidth: float = 0.5):
    """
    Standalone mesh-plotting function.
    Always creates its own figure and axes.

    Parameters
    ----------
    mesh : Mesh
        Must provide mesh.points (N,2), mesh.cells (M,k), mesh.ele_type.
    linewidth : float
        Line width for element edges.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """

    xy = mesh.points[:, :2]

    element_edges = {
        "QUAD4": [(0, 1), (1, 2), (2, 3), (3, 0)],
    }

    ele_type = mesh.ele_type.upper()
    if ele_type not in element_edges:
        raise ValueError(
            f"Unsupported ele_type '{mesh.ele_type}'. "
            f"Supported: {list(element_edges.keys())}"
        )

    edges = element_edges[ele_type]

    segments = []
    for cell in mesh.cells:
        for a, b in edges:
            i = int(cell[a])
            j = int(cell[b])
            segments.append([xy[i], xy[j]])

    fig, ax = plt.subplots(figsize=(6, 6))

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    lc = LineCollection(segments, colors="k", linewidths=linewidth)
    ax.add_collection(lc)

    ax.set_xlim(xy[:, 0].min(), xy[:, 0].max())
    ax.set_ylim(xy[:, 1].min(), xy[:, 1].max())
    ax.set_aspect("equal")

    ax.set_xticks([])
    ax.set_yticks([])

    return fig, ax


def show_rho_pages(rho_list, titles, Nx, Ny, per_page=6, cmap="gray_r"):
    """
    Paginated viewer for topology optimization density fields.

    Arrow keys:
        → next page
        ← previous page
    """

    # Pre-reshape all rho fields once
    images = [jnp.reshape(jnp.asarray(rho), (Ny, Nx), order="F") for rho in rho_list]

    total = len(images)
    pages = (total + per_page - 1) // per_page
    current_page = 0

    fig = plt.figure(figsize=(10, 6))

    def draw_page(page_idx):
        plt.clf()

        start = page_idx * per_page
        end = min(start + per_page, total)
        n = end - start

        ncols = 3
        nrows = int(jnp.ceil(n / ncols))

        for i in range(n):
            ax = plt.subplot(nrows, ncols, i + 1)
            ax.imshow(images[start + i], cmap=cmap, origin="lower", vmin=0, vmax=1)
            ax.set_title(titles[start + i], fontsize=10)
            ax.axis("off")

        plt.suptitle(f"Page {page_idx + 1}/{pages}  (← / →)", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.draw()

    def on_key(event):
        nonlocal current_page
        if event.key == "right":
            current_page = (current_page + 1) % pages
        elif event.key == "left":
            current_page = (current_page - 1) % pages
        draw_page(current_page)

    fig.canvas.mpl_connect("key_press_event", on_key)

    draw_page(0)
    plt.show()


def save_rho_png(rho, title, Nx, Ny, path, cmap="gray_r", dpi=300):
    """
    Save a single topology optimization density field as a PNG.

    Parameters
    ----------
    rho : array-like
        Density field (flattened or array-like).
    title : str
        Title for the figure.
    Nx, Ny : int
        Grid dimensions.
    path : str
        Output file path (e.g., "result.png").
    cmap : str, optional
        Matplotlib colormap.
    dpi : int, optional
        Resolution of saved image.
    """

    rho_img = jnp.reshape(jnp.asarray(rho), (Ny, Nx), order="F")

    plt.figure(figsize=(5, 5))
    plt.imshow(rho_img, cmap=cmap, origin="lower", vmin=0, vmax=1)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def evaluate_models(
    *,
    save_dir: Path,
    scale: Optional[int] = None,
    visualize: bool = False,
) -> pd.DataFrame:
    """
    Evaluate and optionally visualize models saved by serialize_ensemble.

    Args:
        save_dir: directory with saved models
        scale: optional override for resolution scaling factor
        visualize: whether to show rho pages (default: False)

    Returns:
        Pandas DataFrame with per-model metrics
    """
    if not save_dir.exists():
        raise FileNotFoundError(f"SAVE_DIR not found: {save_dir}")

    train_config_path = save_dir / "training_config_snapshot.yaml"
    train_config: TrainingConfig = TrainingConfig.from_yaml(train_config_path)

    Lx = float(train_config.training.Lx)
    Ly = float(train_config.training.Ly)

    scale_original = int(train_config.training.scale)
    used_scale = int(scale) if scale is not None else scale_original

    Nx, Ny = int(Lx * used_scale), int(Ly * used_scale)
    ele_type = "QUAD4"

    print(f"Domain: Lx={Lx}, Ly={Ly} | scale={used_scale} -> Nx={Nx}, Ny={Ny}")

    config_files = sorted(save_dir.glob("*_config.yaml"))
    if not config_files:
        raise RuntimeError(f"No per-model config files found in {save_dir}")

    mesh = rectangle_mesh(Nx=Nx, Ny=Ny, domain_x=Lx, domain_y=Ly)
    geom = get_element_geometry(mesh)
    coords = geom["centroids_scaled"]

    fixed_location, load_location = make_bc_preset("cantilever_corner", Lx, Ly)

    solve_forward, evaluate_volume, filter_fn, _, num_nodes = create_objective_functions(
        mesh,
        fixed_location,
        load_location,
        ele_type=ele_type,
        check_convergence=True,
        verbose=False,
        radius=train_config.training.helmholtz_radius,
        fwd_linear_solver="spsolve",
        bwd_linear_solver="spsolve",
    )
    if train_config.training.helmholtz_radius:
        Ny += 1
        Nx += 1

    print(f"Evaluating with {num_nodes} design variables")
    complience_timer = StepTimer()

    images, titles = [], []
    records = []

    def make_title_from_record(record: dict) -> str:
        lines = [record["model"]]

        compliance = record["compliance"]
        rho_actual = record["rho_actual"]
        rho_error = record.get("rho_error")

        lines.append(f"C={compliance:.3f} | ρ={rho_actual:.3f} (Δ={rho_error:.3f})")

        extras = []
        if record.get("penalty") is not None:
            extras.append(f"pen={record['penalty']:g}")
        if record.get("omega") is not None:
            extras.append(f"ω={record['omega']:g}")

        if extras:
            lines.append(" | ".join(extras))

        return "\n".join(lines)

    for cfg_path in config_files:
        model, _, _, cfg = load_model_from_config(cfg_path, save_dir)

        rho_raw = sigmoid(model(coords))
        rho_filtered = filter_fn(rho_raw)
        rho_pred = jnp.reshape(rho_filtered, (Ny, Nx), order="F")
        

        complience_timer.start()
        compliance = float(solve_forward(rho_filtered))
        solve_time = complience_timer.stop()
        print(f"Solve took {solve_time:3f}s")

        rho_actual = float(evaluate_volume(rho_filtered))

        training = cfg.get("training", {})
        rho_target = training.get("target_density")
        penalty = training.get("penalty")

        model_kwargs = cfg.get("model_kwargs", {}) if isinstance(cfg, dict) else {}
        omega = model_kwargs.get("omega")

        weights_file = cfg.get("weights_file")
        name = Path(weights_file).stem if weights_file else cfg_path.stem

        record = dict(
            model=name,
            compliance=compliance,
            rho_actual=rho_actual,
            rho_target=rho_target,
            rho_error=(
                abs(rho_actual - rho_target) if rho_target is not None else None
            ),
            penalty=penalty,
            omega=omega,
            scale=scale_original,
        )
        records.append(record)

        title = make_title_from_record(record)
        titles.append(title)
        images.append(rho_pred)

        out_path = save_dir / f"{name}_rho.png"
        save_rho_png(rho_pred, title, Nx, Ny, out_path)

    df = pd.DataFrame.from_records(records).set_index("model")

    df_out = df.sort_values("compliance").round(
        {
            "compliance": 4,
            "rho_actual": 4,
            "rho_target": 4,
            "rho_error": 4,
        }
    )

    print("\nModel evaluation summary:\n")
    print(df_out.to_string())

    csv_path = save_dir / "model_evaluation.csv"
    df_out.to_csv(csv_path)
    print(f"\nSaved evaluation table to: {csv_path}")

    if visualize:
        show_rho_pages(images, titles, Nx=Nx, Ny=Ny, per_page=6)

    return df_out
