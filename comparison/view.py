import pyvista as pv
import argparse

def plot_vtk(path: str):
    mesh = pv.read(path)

    plotter = pv.Plotter()

    # Add the displacement field (vector)
    plotter.add_mesh(
        mesh,
        scalars="displacement",   # Use the vector field stored in VTK
        show_edges=False,
        cmap="coolwarm",
        render_points_as_spheres=False
    )

    # Add interactive arrows to show displacement vectors
    plotter.add_arrows(
        mesh.points,
        mesh["displacement"],
        mag=1.0,    # scaling factor
        opacity=0.7
    )

    plotter.view_xz()
    plotter.add_axes()
    plotter.show_grid()
    plotter.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot VTU file with displacement.")
    parser.add_argument("path", type=str, help="Path to the .vtu file")
    args = parser.parse_args()

    plot_vtk(args.path)