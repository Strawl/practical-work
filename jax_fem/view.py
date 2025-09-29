import pyvista as pv

mesh = pv.read("./data/vtk/sol_050.vtu")

plotter = pv.Plotter()
plotter.add_mesh(
    mesh,
    scalars="theta",
    cmap="gray_r",
    show_edges=False,
)
plotter.view_xy()
plotter.remove_bounds_axes()
plotter.show_bounds(show_zaxis=False)
plotter.show()