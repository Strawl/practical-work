from typing import Optional, Tuple

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from feax.mesh import Mesh


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

    xy = mesh.points[:, :2]  # first two dimensions only

    # Supported element edge definitions
    element_edges = {
        'TRI3':  [(0, 1), (1, 2), (2, 0)],
        'TRI6':  [(0, 1), (1, 2), (2, 0)],   # ignore midside nodes
        'QUAD4': [(0, 1), (1, 2), (2, 3), (3, 0)],
        'QUAD8': [(0, 1), (1, 2), (2, 3), (3, 0)],

        'TET4':  [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)],
        'TET10': [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)],

        'HEX8':  [(0, 1), (1, 2), (2, 3), (3, 0),
                  (4, 5), (5, 6), (6, 7), (7, 4),
                  (0, 4), (1, 5), (2, 6), (3, 7)],
        'HEX20': [(0, 1), (1, 2), (2, 3), (3, 0),
                  (4, 5), (5, 6), (6, 7), (7, 4),
                  (0, 4), (1, 5), (2, 6), (3, 7)],
        'HEX27': [(0, 1), (1, 2), (2, 3), (3, 0),
                  (4, 5), (5, 6), (6, 7), (7, 4),
                  (0, 4), (1, 5), (2, 6), (3, 7)],
    }

    ele_type = mesh.ele_type.upper()
    if ele_type not in element_edges:
        raise ValueError(
            f"Unsupported ele_type '{mesh.ele_type}'. "
            f"Supported: {list(element_edges.keys())}"
        )

    edges = element_edges[ele_type]

    # Build line segments for LineCollection
    segments = []
    for cell in mesh.cells:
        for a, b in edges:
            i = int(cell[a])
            j = int(cell[b])
            segments.append([xy[i], xy[j]])

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(6, 6))

    # White background
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Draw edge segments
    lc = LineCollection(segments, colors="k", linewidths=linewidth)
    ax.add_collection(lc)

    # Bounds
    ax.set_xlim(xy[:, 0].min(), xy[:, 0].max())
    ax.set_ylim(xy[:, 1].min(), xy[:, 1].max())
    ax.set_aspect("equal")

    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    return fig, ax