from pathlib import Path
import argparse

import pyvista as pv


def parse_args():
    parser = argparse.ArgumentParser(description="Render a VTU file with PyVista.")
    parser.add_argument("path", type=Path, help="Path to the .vtu file")
    parser.add_argument(
        "--scalars",
        type=str,
        default=None,
        help="Point or cell field to color by. Defaults to the first available field.",
    )
    parser.add_argument(
        "--show-edges",
        action="store_true",
        help="Render mesh edges.",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="gray_r",
        help="Matplotlib/PyVista colormap name.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    mesh = pv.read(args.path)

    available = list(mesh.point_data.keys()) + list(mesh.cell_data.keys())
    if not available:
        raise ValueError(f"No scalar fields found in {args.path}")

    scalars = args.scalars or available[0]
    if scalars not in available:
        raise ValueError(
            f"Scalar field '{scalars}' not found. Available fields: {available}"
        )

    plotter = pv.Plotter()
    plotter.add_mesh(
        mesh,
        scalars=scalars,
        cmap=args.cmap,
        show_edges=args.show_edges,
    )
    plotter.view_xy()
    plotter.remove_bounds_axes()
    plotter.show_bounds(show_zaxis=False)
    plotter.show()


if __name__ == "__main__":
    main()
