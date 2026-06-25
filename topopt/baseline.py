from __future__ import annotations

from pathlib import Path

import feax.gene as gene
import jax.numpy as jnp
import numpy as np
from feax.gene.optimizer import Pipeline, constraint
from feax.mesh import rectangle_mesh

from topopt.bc import (
    equivalent_traction_for_point_load,
    make_dirichlet_boundary_conditions,
    make_neumann_boundary_location,
    make_neumann_surface_var_fn,
)
from topopt.evaluation import save_rho_png
from topopt.fem_utils import create_objective_functions, create_surface_vars
from topopt.monitoring import MetricTracker


def _save_history_metrics(save_dir: Path, history: dict[str, list[float]]) -> None:
    tracker = MetricTracker(save_dir=save_dir, fill_invalid=True)

    for value in history.get("objective", ()):
        tracker.log("compliance", value)
    for value in history.get("volume", ()):
        tracker.log("volume", value)

    if tracker.data:
        tracker.save()
        tracker.plot_all_metrics_across_models(
            model_names=["Baseline"],
            save=True,
            show=False,
        )


def run_feax_topopt_mma(
    Lx: int,
    Ly: int,
    save_dir: Path,
    scale: float = 1.0,
    dirichlet_boundary_conditions: str = "cantilever_left_support",
    neumann_boundary_conditions: str = "cantilever_corner",
    vol_frac: float = 0.5,
    ele_type: str = "QUAD4",
    E0: float = 1.0,
    E_eps: float = 1e-6,
    nu: float = 0.3,
    p: float = 3.0,
    T: float = 1e2,
    problem_type: str = "density",
    load_mode: str = "traction",
    point_load_magnitude: float = 1.0,
    gauss_order: int = 2,
    iter_num: int = 1,
    max_iter: int = 100,
    radius: float = 0.3,
    heaviside_beta: float = 10.0,
    heaviside_threshold: float = 0.5,
    solver: str = "cholmod",
    save_every: int = 5,
):
    if not (0.0 < vol_frac <= 1.0):
        raise ValueError(f"vol_frac must be in (0, 1], got {vol_frac}")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    Nx = int(Lx * scale)
    Ny = int(Ly * scale)
    mesh = rectangle_mesh(Nx, Ny, domain_x=Lx, domain_y=Ly)
    rho_init = np.full(mesh.points.shape[0], vol_frac, dtype=float)

    problem_type = str(problem_type).strip().lower()
    if problem_type not in {"density", "plane_stress"}:
        raise ValueError(
            "problem_type must be 'density' or 'plane_stress', "
            f"got {problem_type!r}"
        )

    load_mode = str(load_mode).strip().lower()
    if load_mode not in {"traction", "equivalent_point_load"}:
        raise ValueError(
            "load_mode must be 'traction' or 'equivalent_point_load', "
            f"got {load_mode!r}"
        )

    if load_mode == "equivalent_point_load":
        traction_value = equivalent_traction_for_point_load(
            neumann_boundary_conditions,
            point_load_magnitude=point_load_magnitude,
            Ly=Ly,
            Ny=Ny,
        )
    else:
        traction_value = T

    dirichlet_boundary_condition_specs = make_dirichlet_boundary_conditions(
        dirichlet_boundary_conditions,
        Lx,
        Ly,
    )
    neumann_boundary_location = make_neumann_boundary_location(Lx, Ly)
    surface_var_fn = make_neumann_surface_var_fn(
        neumann_boundary_conditions,
        Lx,
        Ly,
        traction_value=traction_value,
        Ny=Ny,
    )

    class TopOpt2DPipeline(Pipeline):
        def _apply_filter_only(self, rho):
            return self._filter_fn(rho)

        def _apply_density_pipeline(self, rho):
            rho_filtered = self._apply_filter_only(rho)
            if heaviside_beta > 0.0:
                return gene.heaviside_projection(
                    rho_filtered,
                    beta=heaviside_beta,
                    threshold=heaviside_threshold,
                )
            return rho_filtered

        def build(self, mesh):
            (
                self._solve_forward,
                self._evaluate_volume,
                self._filter_fn,
                _,
                _,
                self._problem,
            ) = create_objective_functions(
                mesh=mesh,
                dirichlet_boundary_conditions=dirichlet_boundary_condition_specs,
                neumann_boundary_location=neumann_boundary_location,
                target_fraction=vol_frac,
                ele_type=ele_type,
                E0=E0,
                E_eps=E_eps,
                nu=nu,
                p=p,
                T=traction_value,
                gauss_order=gauss_order,
                iter_num=iter_num,
                check_convergence=True,
                verbose=False,
                radius=radius,
                fwd_linear_solver=solver,
                bwd_linear_solver=solver,
                problem_type=problem_type,
            )
            self._surface_vars = create_surface_vars(
                self._problem,
                (surface_var_fn,),
            )[0]

        def objective(self, rho, **_params):
            rho_projected = self._apply_density_pipeline(rho)
            return self._solve_forward(rho_projected, self._surface_vars)

        @constraint(target=vol_frac)
        def volume(self, rho, **_params):
            rho_projected = self._apply_density_pipeline(rho)
            return self._evaluate_volume(rho_projected)

        def filter(self, rho):
            return self._apply_density_pipeline(rho)

        def filtered_density(self, rho):
            return self._apply_filter_only(rho)

        def projected_density(self, rho):
            return self._apply_density_pipeline(rho)

        def compliance_for_density(self, rho):
            return self._solve_forward(rho, self._surface_vars)

        def volume_fraction_for_density(self, rho):
            return self._evaluate_volume(rho)

    pipeline = TopOpt2DPipeline()

    result = gene.optimizer.run(
        pipeline=pipeline,
        mesh=mesh,
        max_iter=max_iter,
        output_dir=str(save_dir),
        save_every=save_every,
        rho_init=rho_init,
        rho_bounds=(0.001, 1.0),
        jit=True,
    )

    x_opt_unfiltered = jnp.asarray(result.rho)
    x_opt_filtered = jnp.asarray(pipeline.filtered_density(x_opt_unfiltered))
    x_opt_projected = jnp.asarray(pipeline.projected_density(x_opt_unfiltered))

    compliance_unfiltered = float(pipeline.compliance_for_density(x_opt_unfiltered))
    compliance_filtered = float(pipeline.compliance_for_density(x_opt_filtered))
    compliance_projected = float(pipeline.compliance_for_density(x_opt_projected))
    volume_unfiltered = float(pipeline.volume_fraction_for_density(x_opt_unfiltered))
    volume_filtered = float(pipeline.volume_fraction_for_density(x_opt_filtered))
    volume_projected = float(pipeline.volume_fraction_for_density(x_opt_projected))

    print("-" * 60)
    print("2D topology optimization finished with feax.gene pipeline")
    print(
        f"Problem setup: problem_type={problem_type}, load_mode={load_mode}, "
        f"traction={traction_value:.6g}"
    )
    if heaviside_beta > 0.0:
        print(
            "Final compliance\n"
            f"  unfiltered: {compliance_unfiltered:.6f}\n"
            f"  filtered  : {compliance_filtered:.6f}\n"
            f"  projected : {compliance_projected:.6f}"
        )
        print(
            "Final volume fraction\n"
            f"  unfiltered: {volume_unfiltered:.6f}\n"
            f"  filtered  : {volume_filtered:.6f}\n"
            f"  projected : {volume_projected:.6f}\n"
            f"  target    : {vol_frac:.6f}"
        )
    else:
        print(
            "Final compliance\n"
            f"  unfiltered: {compliance_unfiltered:.6f}\n"
            f"  filtered  : {compliance_filtered:.6f}"
        )
        print(
            "Final volume fraction\n"
            f"  unfiltered: {volume_unfiltered:.6f}\n"
            f"  filtered  : {volume_filtered:.6f}\n"
            f"  target    : {vol_frac:.6f}"
        )

    save_rho_png(
        x_opt_unfiltered,
        "Final (unfiltered)",
        Nx=Nx + 1,
        Ny=Ny + 1,
        path=save_dir / "rho_final_unfiltered.png",
    )
    save_rho_png(
        x_opt_filtered,
        "Final (filtered)",
        Nx=Nx + 1,
        Ny=Ny + 1,
        path=save_dir / "rho_final_filtered.png",
    )
    save_rho_png(
        x_opt_projected,
        "Final (projected)",
        Nx=Nx + 1,
        Ny=Ny + 1,
        path=save_dir / "rho_final_projected.png",
    )
    save_rho_png(
        x_opt_unfiltered,
        "Final",
        Nx=Nx + 1,
        Ny=Ny + 1,
        path=save_dir / "rho_final.png",
    )

    np.savez_compressed(
        save_dir / "baseline_final_with_without_filter.npz",
        rho_unfiltered=np.asarray(x_opt_unfiltered),
        rho_filtered=np.asarray(x_opt_filtered),
        rho_projected=np.asarray(x_opt_projected),
        compliance_unfiltered=compliance_unfiltered,
        compliance_filtered=compliance_filtered,
        compliance_projected=compliance_projected,
        volume_fraction_unfiltered=volume_unfiltered,
        volume_fraction_filtered=volume_filtered,
        volume_fraction_projected=volume_projected,
        target_volume_fraction=float(vol_frac),
        problem_type=problem_type,
        load_mode=load_mode,
        traction=float(traction_value),
        point_load_magnitude=float(point_load_magnitude),
        heaviside_beta=float(heaviside_beta),
        heaviside_threshold=float(heaviside_threshold),
        **{
            f"history_{name}": np.asarray(values)
            for name, values in sorted(result.history.items())
        },
    )

    _save_history_metrics(save_dir, result.history)
    return result
