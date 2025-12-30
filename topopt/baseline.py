# baseline.py
import os

import config
import jax.numpy as jnp
import nlopt
from bc import make_bc_preset
from feax.mesh import rectangle_mesh
from fem_utils import create_objective_functions
from monitoring import MetricTracker
from visualize import save_rho_png

import jax


def run_feax_topopt_mma(
    Lx: int,
    Ly: int,
    scale: float = 1.0,
    bc_preset_name: str = "cantilever_corner",
    vol_frac: float = 0.5,
    ele_type: str = "QUAD4",
    E0: float = 70e3,
    E_eps: float = 7.0,
    nu: float = 0.3,
    p: float = 3.0,
    T: float = 1e2,
    gauss_order: int = 2,
    iter_num: int = 1,
    max_iter: int = 500,
    radius: float = 1.0,
    print_every: int = 5,
    save_every: int = 5,
):
    Nx = int(Lx * scale)
    Ny = int(Ly * scale)
    mesh = rectangle_mesh(Nx, Ny, domain_x=Lx, domain_y=Ly)

    fixed_location, load_location = make_bc_preset(bc_preset_name, Lx, Ly)

    solve_forward, evaluate_volume, rho_init, num_nodes = create_objective_functions(
        mesh=mesh,
        fixed_location=fixed_location,
        load_location=load_location,
        target_fraction=vol_frac,
        ele_type=ele_type,
        E0=E0,
        E_eps=E_eps,
        nu=nu,
        p=p,
        T=T,
        gauss_order=gauss_order,
        iter_num=iter_num,
        check_convergence=True,
        verbose=False,
        radius=radius,
    )

    forward_jit = jax.jit(solve_forward)
    volume_jit = jax.jit(evaluate_volume)
    grad_complience_jit = jax.jit(jax.grad(forward_jit))
    grad_volume_jit = jax.jit(jax.grad(volume_jit))

    tracker = MetricTracker(output_dir=config.SAVE_DIR, fill_invalid=True)
    iteration_count = [0]

    def objective(x, grad):
        rho = jnp.array(x)

        f = float(forward_jit(rho))
        grad[:] = jnp.array(grad_complience_jit(rho))
        v = float(volume_jit(rho))

        iteration_count[0] += 1
        tracker.log("compliance", f)
        tracker.log("volume", v)

        if iteration_count[0] % print_every == 0:
            print(f"Iter {iteration_count[0]:4d}: Complience={f:.6f}, Volume={v:.4f}")

        if iteration_count[0] % save_every == 0:
            save_rho_png(
                jnp.array(rho),
                f"{iteration_count[0]}",
                Nx=Nx + 1,
                Ny=Ny + 1,
                path=os.path.join(config.SAVE_DIR, f"rho_{iteration_count[0]}.png"),
            )
            tracker.save()

        return f

    def volume_constraint(x, grad):
        v = float(volume_jit(x))
        grad[:] = jnp.array(grad_volume_jit(x))
        return v - vol_frac

    opt = nlopt.opt(nlopt.LD_MMA, num_nodes)
    opt.set_lower_bounds(0.001)
    opt.set_upper_bounds(1.0)
    opt.set_min_objective(objective)
    opt.add_inequality_constraint(volume_constraint, 1e-8)
    opt.set_maxeval(max_iter)

    print("Starting topology optimization with NLopt MMA...")
    print(f"Number of design variables: {num_nodes}")
    print(f"Target volume fraction: {vol_frac}")
    print(f"Running with the boundary conditions: {bc_preset_name}")
    print(f"Shape: Nx={Nx}, Ny={Ny}")
    print("-" * 60)
    x0 = jnp.array(rho_init)
    try:
        x_opt = opt.optimize(x0)
        opt_val = opt.last_optimum_value()
    except nlopt.RoundoffLimited:
        print("Optimization stopped due to roundoff errors (converged)")
        x_opt = x0
        opt_val = float(forward_jit(x_opt))

    print("-" * 60)
    print("Optimization finished!")
    print(f"Final compliance: {opt_val:.4e}")
    print(f"Final volume fraction: {float(volume_jit(x_opt)):.4f}")

    save_rho_png(
        x_opt,
        "Final",
        Nx=Nx + 1,
        Ny=Ny + 1,
        path=config.SAVE_DIR / "rho_final.png",
    )
    tracker.save()
    tracker.plot_all_metrics_across_models(
        model_names=["Baseline"], save=True, show=False
    )


if __name__ == "__main__":
    run_feax_topopt_mma(
        Lx=60,
        Ly=30,
        scale=1,
        bc_preset_name="cantilever_corner",
        vol_frac=0.5,
        radius=0.8,
        max_iter=10,
        save_every=5,
        print_every=1,
    )
