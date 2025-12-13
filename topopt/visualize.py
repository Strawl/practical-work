
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from feax.mesh import Mesh
import jax.numpy as np


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
        'QUAD4': [(0, 1), (1, 2), (2, 3), (3, 0)],
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


def show_rho_pages(
    rho_list,
    titles,
    Nx,
    Ny,
    per_page=6,
    cmap="gray_r"
):
    """
    Paginated viewer for topology optimization density fields.

    Arrow keys:
        → next page
        ← previous page
    """

    # Pre-reshape all rho fields once
    images = [
        np.reshape(np.asarray(rho), (Ny, Nx), order="F")
        for rho in rho_list
    ]

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
        nrows = int(np.ceil(n / ncols))

        for i in range(n):
            ax = plt.subplot(nrows, ncols, i + 1)
            ax.imshow(images[start + i], cmap=cmap, origin="lower", vmin=0, vmax=1)
            ax.set_title(titles[start + i], fontsize=10)
            ax.axis("off")

        plt.suptitle(
            f"Page {page_idx + 1}/{pages}  (← / →)",
            fontsize=14
        )
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
