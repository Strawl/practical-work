import argparse
import logging
from pathlib import Path

import jax.numpy as np
from jax_fem.generate_mesh import Mesh, get_meshio_cell_type, rectangle_mesh
from jax_fem.problem import Problem
from jax_fem.solver import ad_wrapper
from serialization import load_model_from_config

import jax

logging.getLogger("jax_fem").setLevel(logging.WARNING)


# -----------------------------
# Define elasticity problem
# -----------------------------
class Elasticity(Problem):
    def custom_init(self):
        self.fe = self.fes[0]
        # all elements are "flexible" (design variables)
        self.fe.flex_inds = np.arange(len(self.fe.cells))

    def get_tensor_map(self):
        def stress(u_grad, theta):
            # Plane stress elasticity
            Emax = 70.0e3
            Emin = 1e-3 * Emax
            nu = 0.3
            penal = 3.0
            E = Emin + (Emax - Emin) * theta[0] ** penal

            epsilon = 0.5 * (u_grad + u_grad.T)
            eps11, eps22, eps12 = epsilon[0, 0], epsilon[1, 1], epsilon[0, 1]

            sig11 = E / (1 + nu) / (1 - nu) * (eps11 + nu * eps22)
            sig22 = E / (1 + nu) / (1 - nu) * (nu * eps11 + eps22)
            sig12 = E / (1 + nu) * eps12
            return np.array([[sig11, sig12], [sig12, sig22]])

        return stress

    def set_params(self, params):
        # params: (num_flex, n_param_per_elem)
        full_params = np.ones((self.fe.num_cells, params.shape[1]))
        full_params = full_params.at[self.fe.flex_inds].set(params)
        thetas = np.repeat(full_params[:, None, :], self.fe.num_quads, axis=1)
        self.full_params = full_params
        self.internal_vars = [thetas]

    def compute_compliance(self, sol):
        boundary_inds = self.boundary_inds_list[0]
        _, nanson_scale = self.fe.get_face_shape_grads(boundary_inds)

        u_face = (
            sol[self.fe.cells][boundary_inds[:, 0]][:, None, :, :]
            * self.fe.face_shape_vals[boundary_inds[:, 1]][:, :, :, None]
        )
        u_face = np.sum(u_face, axis=2)

        subset_quad_points = self.physical_surface_quad_points[0]
        neumann_fn = self.get_surface_maps()[0]
        traction = -jax.vmap(jax.vmap(neumann_fn))(u_face, subset_quad_points)
        return np.sum(traction * u_face * nanson_scale[:, :, None])

    def get_surface_maps(self):
        def surface_map(u, x):
            # Constant vertical load on the right edge
            return np.array([0.0, 100.0])

        return [surface_map]


# -----------------------------
# Helper: element centroids
# -----------------------------
def get_element_centroids(mesh):
    pts = np.array(mesh.points)
    cells = np.array(mesh.cells)
    centroids = np.mean(pts[cells], axis=1)  # (num_cells, 2)

    # normalize to [-1, 1] for SIREN stability
    xmin, ymin = np.min(centroids, axis=0)
    xmax, ymax = np.max(centroids, axis=0)
    centroids = (centroids - np.array([xmin, ymin])) / (
        np.array([xmax - xmin, ymax - ymin])
    )
    centroids = 2.0 * centroids - 1.0

    return centroids.astype(np.float32)


# -----------------------------
# Compliance via model
# -----------------------------
def calculate_compliance(model, coords, problem, fwd_pred):
    rho = jax.nn.sigmoid(model(coords))  # (num_cells, ?)
    print("rho shape:", rho.shape)

    def J_total(params):
        sol_list = fwd_pred(params)
        return problem.compute_compliance(sol_list[0])

    J = J_total(rho)
    return J


# -----------------------------
# Geometry and BCs
# -----------------------------
ele_type = "QUAD4"
cell_type = get_meshio_cell_type(ele_type)
Lx, Ly = 60.0, 30.0


def fixed_location(point):
    return np.isclose(point[0], 0.0, atol=1e-5)


def load_location(point):
    return np.logical_and(
        np.isclose(point[0], Lx, atol=1e-5),
        np.isclose(point[1], 0.0, atol=0.1 * Ly + 1e-5),
    )


def dirichlet_val(point):
    return 0.0


dirichlet_bc_info = [[fixed_location] * 2, [0, 1], [dirichlet_val] * 2]
location_fns = [load_location]

solver_options = {"umfpack_solver": {}, "tol": 1e-8, "precond": True}


# -----------------------------
# Main: loop over all models at scale 25
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compute compliance of all models at a fixed scale."
    )
    parser.add_argument(
        "--dir",
        type=str,
        help="Directory containing model files + *_config.json.",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=25,  # you asked specifically for scale 25
        help="Scaling factor for resolution (default: 25).",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="60,30",
        help="Domain size as Lx,Ly (default: 60,30).",
    )
    args = parser.parse_args()

    # Determine base directory
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
    print(f"Using scale={scale} -> Nx={Nx}, Ny={Ny}")

    config_files = sorted(base_dir.glob("*_config.json"))
    if not config_files:
        print(f"No per-model config files (*_config.json) found in {base_dir}")
        return

    print(f"Found {len(config_files)} model configs in {base_dir}")

    compliances = []

    # Build mesh once for the chosen scale
    meshio_mesh = rectangle_mesh(Nx=Nx, Ny=Ny, domain_x=Lx, domain_y=Ly)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

    # Set up problem and solver for this mesh
    problem = Elasticity(
        mesh,
        vec=2,
        dim=2,
        ele_type=ele_type,
        dirichlet_bc_info=dirichlet_bc_info,
        location_fns=location_fns,
    )

    fwd_pred = ad_wrapper(
        problem,
        solver_options=solver_options,
        adjoint_solver_options=solver_options,
    )

    # element centroids as NN inputs
    coords = get_element_centroids(mesh)

    for cfg_path in config_files:
        print(f"\nLoading from {cfg_path.name} ...")
        model, _, _, _ = load_model_from_config(cfg_path, base_dir)
        J = calculate_compliance(model, coords, problem, fwd_pred)
        J_val = float(J)
        compliances.append((cfg_path.name, J_val))
        print(f"Compliance (scale {scale}) for {cfg_path.name}: {J_val:.6e}")

    # Summary
    print("\n=== Compliance summary (scale = {}) ===".format(scale))
    for name, J in compliances:
        print(f"{name}: {J}")


if __name__ == "__main__":
    main()
