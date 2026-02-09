# baseline.py
import os
from pathlib import Path

import jax.numpy as jnp
import nlopt
import numpy as np
from feax.mesh import rectangle_mesh

import jax
from topopt.bc import make_bc_preset
from topopt.evaluation import save_rho_png
from topopt.fem_utils import create_objective_functions
from topopt.monitoring import MetricTracker, StepTimer


def run_feax_topopt_mma(
    Lx: int,
    Ly: int,
    save_dir: Path,
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
    max_iter: int = 100,
    radius: float = 0.1,
    print_every: int = 5,
    save_every: int = 5,
):
    step_timer = StepTimer()
    wal_timer = StepTimer()
    Nx = int(Lx * scale)
    Ny = int(Ly * scale)
    mesh = rectangle_mesh(Nx, Ny, domain_x=Lx, domain_y=Ly)

    fixed_location, load_location = make_bc_preset(bc_preset_name, Lx, Ly)

    solve_forward, evaluate_volume, filter_fn, rho_init, num_nodes = create_objective_functions(
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
        fwd_linear_solver="bicgstab",
        bwd_linear_solver="bicgstab",
    )

    forward_jit = jax.jit(solve_forward)
    volume_jit = jax.jit(evaluate_volume)
    grad_complience_jit = jax.jit(jax.grad(forward_jit))
    grad_volume_jit = jax.jit(jax.grad(volume_jit))

    tracker = MetricTracker(save_dir=save_dir, fill_invalid=True)
    o_iter_count = [0]
    v_iter_count = [0]
    compile_time = [0]

    def objective(x, grad):
        rho = filter_fn(x)
        step_timer.start()
        f = float(forward_jit(rho))
        grad[:] = jnp.array(grad_complience_jit(rho))
        jax.block_until_ready(f)
        jax.block_until_ready(grad[:])
        step_time_s = step_timer.stop()

        tracker.log("compliance", f)

        if o_iter_count[0] > 1:
            tracker.log("objective_wall_time_s", step_time_s)
        else:
            compile_time[0] += step_time_s

        if o_iter_count[0] % print_every == 0:
            print(f"Iter {o_iter_count[0]:4d}: Complience={f:.6f}")

        if o_iter_count[0] % save_every == 0:
            save_rho_png(
                jnp.array(rho),
                f"{o_iter_count[0]}",
                Nx=Nx + 1,
                Ny=Ny + 1,
                path=os.path.join(save_dir, f"rho_{o_iter_count[0]}.png"),
            )
            tracker.save()

        o_iter_count[0] += 1
        return f

    def volume_constraint(x, grad):
        rho = filter_fn(x)

        step_timer.start()
        v = float(volume_jit(rho))
        grad[:] = jnp.array(grad_volume_jit(rho))
        jax.block_until_ready(v)
        jax.block_until_ready(grad[:])
        step_time_s = step_timer.stop()

        tracker.log("volume", v)

        if v_iter_count[0] > 1:
            tracker.log("volume_constraint_wall_time_s", step_time_s)
        else:
            compile_time[0] += step_time_s

        if v_iter_count[0] % print_every == 0:
            print(f"Volume={v:.4f}")

        v_iter_count[0] += 1
        return v - vol_frac

    opt = nlopt.opt(nlopt.LD_MMA, num_nodes)
    opt.set_lower_bounds(0.001)
    opt.set_upper_bounds(1.0)
    opt.set_min_objective(objective)
    opt.add_inequality_constraint(volume_constraint, 1e-4)
    opt.set_maxeval(max_iter)
    opt.set_maxtime(3600)

    print("Starting topology optimization with NLopt MMA...")
    print(f"Number of design variables: {num_nodes}")
    print(f"Target volume fraction: {vol_frac}")
    print(f"Running with the boundary conditions: {bc_preset_name}")
    print(f"Shape: Nx={Nx}, Ny={Ny}")
    print("-" * 60)
    x0 = jnp.array(rho_init)
    try:
        wal_timer.start()
        x_opt = opt.optimize(x0)
    except nlopt.RoundoffLimited:
        print("Optimization stopped due to roundoff errors (converged)")
        x_opt = x0

    wal_time = wal_timer.stop()
    x_opt_unfiltered = jnp.asarray(x_opt)
    x_opt_filtered = jnp.asarray(filter_fn(x_opt_unfiltered))

    compliance_unfiltered = float(forward_jit(x_opt_unfiltered))
    compliance_filtered = float(forward_jit(x_opt_filtered))
    volume_unfiltered = float(volume_jit(x_opt_unfiltered))
    volume_filtered = float(volume_jit(x_opt_filtered))

    # Objective + volume wall times (steady-state only)
    obj_hist = tracker.stack("objective_wall_time_s")
    vol_hist = tracker.stack("volume_constraint_wall_time_s")
    hot_time = float(jnp.sum(obj_hist) + jnp.sum(vol_hist))
    other_time = wal_time - hot_time - compile_time[0]
    share_hot = hot_time / wal_time
    share_compile = compile_time[0] / wal_time
    share_other = other_time / wal_time
    print("-" * 60)
    print("Optimization finished!")
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
    print(
        "Timing summary\n"
        f"  wall total    : {wal_time:8.3f}s (100.00%)\n"
        f"  hot optimise  : {hot_time:8.3f}s ({100.0 * share_hot:6.2f}%)\n"
        f"  compile/first : {compile_time[0]:8.3f}s ({100.0 * share_compile:6.2f}%)\n"
        f"  other         : {other_time:8.3f}s ({100.0 * share_other:6.2f}%)"
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
    # Backwards-compatible alias (previously saved the unfiltered rho as rho_final.png).
    save_rho_png(
        x_opt_unfiltered,
        "Final",
        Nx=Nx + 1,
        Ny=Ny + 1,
        path=save_dir / "rho_final.png",
    )

    save_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        save_dir / "baseline_final_with_without_filter.npz",
        rho_unfiltered=np.asarray(x_opt_unfiltered),
        rho_filtered=np.asarray(x_opt_filtered),
        compliance_unfiltered=compliance_unfiltered,
        compliance_filtered=compliance_filtered,
        volume_fraction_unfiltered=volume_unfiltered,
        volume_fraction_filtered=volume_filtered,
        target_volume_fraction=float(vol_frac),
    )
    tracker.save()
    tracker.plot_all_metrics_across_models(
        model_names=["Baseline"], save=True, show=False
    )
