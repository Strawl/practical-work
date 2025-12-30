from datetime import datetime
from pathlib import Path
from typing import Callable
import jax
import jax.numpy as jnp
import optax
import os
from feax.experimental.topopt_toolkit import mdmm
from feax.mesh import rectangle_mesh, Mesh
from sklearn.model_selection import ParameterGrid
import pandas as pd

from bc import make_bc_preset
from fem_utils import create_J_total
import matplotlib.pyplot as plt

from visualize import save_rho_png




def run_topopt_mdmm(
    J_total: Callable[[jnp.ndarray], jnp.ndarray],
    rho_shape,
    vol_frac: float = 0.4,
    num_steps: int = 1000,
    learning_rate: float = 5e-2,
    constraint_damping: float = 1.0,
    constraint_weight: float = 1.0,
    print_every: int = 10,
    vol_tol: float = 1e-3
):
    """
    Minimize compliance subject to a volume-fraction constraint using MDMM.

    Args
    ----
    J_total:
        Callable taking rho (density field) and returning scalar compliance.
    rho_shape:
        Shape tuple for rho, e.g. (n_el,) or (ny, nx).
    vol_frac:
        Target volume fraction V / V0.
    num_steps:
        Number of MDMM optimization iterations.
    learning_rate:
        Base learning rate for the primary variables.
    constraint_damping:
        MDMM damping parameter.
    constraint_weight:
        Weight of the constraint loss relative to primary loss.
    print_every:
        Logging interval.

    Returns
    -------
    rho_opt:
        Optimized density field.
    history:
        Dict with lists of 'J', 'vol', 'infeas'.
    """

    volume_constraint = mdmm.ineq(
        lambda rho: vol_frac - jnp.mean(rho),
        damping=10,
        weight=constraint_weight,
        reduction=jnp.sum,
    )

    initial_rho = jnp.full(rho_shape, vol_frac)
    alpha0 = jnp.log(initial_rho / (1.0 - initial_rho))

    mdmm_params0 = volume_constraint.init(initial_rho)

    params = {
        "alpha": alpha0,
        "mdmm": mdmm_params0,
    }

    def loss_fn(params):
        alpha = params["alpha"]
        mdmm_params = params["mdmm"]

        rho = jax.nn.sigmoid(alpha)
        J = J_total(rho)

        c_loss, infeas = volume_constraint.loss(mdmm_params, rho)
        # c_loss = jnp.maximum(c_loss, 0)
        total_loss = J + c_loss

        aux = {
            "J": J,
            "c_loss": c_loss,
            "vol": jnp.mean(rho),
            "infeas": infeas,
            "rho": rho,
        }
        return total_loss, aux

    PATIENCE = 10
    COOLDOWN = 5
    FACTOR = 0.5
    RTOL = 0.05
    ACCUMULATION_SIZE = 5
    inner_optim = optax.chain(
        optax.adabelief(learning_rate),
        mdmm.optax_prepare_update(),

        optax.contrib.reduce_on_plateau(
            patience=PATIENCE,
            cooldown=COOLDOWN,
            factor=FACTOR,
            rtol=RTOL,
            accumulation_size=ACCUMULATION_SIZE,
        ),
    )
    optimizer = optax.apply_if_finite(
        inner_optim,
        max_consecutive_errors=30,
    )
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state):
        (loss_val, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params, value=loss_val)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val, aux

    J_window = 10
    J_window_tol = 1e-3
    history = {"J": [], "vol": [], "infeas": [], "rho": [], "lr_scales": [], "c_loss": []}
    final_iter = num_steps

    for it in range(num_steps):
        params, opt_state, loss_val, aux = step(params, opt_state)

        J_val = float(aux["J"])
        vol_val = float(aux["vol"])
        infeas_val = float(aux["infeas"])
        lr_scales = optax.tree.get(opt_state, "scale")
        c_loss = float(aux["c_loss"])

        history["J"].append(J_val)
        history["vol"].append(vol_val)
        history["infeas"].append(infeas_val)
        history["rho"].append(aux["rho"])
        history["lr_scales"].append(lr_scales)
        history["c_loss"].append(c_loss)

        if (it % print_every) == 0 or it == num_steps - 1:
            print(
                f"[{it:05d}] "
                f"Loss = {loss_val:.4f}, "
                f"J = {J_val:.4f}, "
                f"C_loss = {c_loss:.4f}, "
                f"Infeas = {infeas_val:.4f}, "
                f"vol = {vol_val:.4f}, "
                f"Lr scales = {lr_scales:.3f}"
            )

        # ---- Windowed stagnation criterion ----
        if it >= J_window:
            J_start = history["J"][it - J_window]
            J_end = history["J"][it]
            J_delta = abs(J_end - J_start)

            vol_close = abs(vol_val - vol_frac) <= vol_tol
            J_delta_rel = abs(J_end - J_start) / (abs(J_start) + 1e-12)

            if vol_close and J_delta_rel <= J_window_tol:
                print(
                    f"Early stopping at iter {it}: "
                    f"|ΔJ| over last {J_window} iters = {J_delta:.3e}"
                )
                final_iter = it + 1
                break


    # Final density field
    rho_opt = jax.nn.sigmoid(params["alpha"])
    return rho_opt, history, final_iter

def run_feax_topopt(
    mesh,
    Lx: float,
    Ly: float,
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
    num_steps: int = 1000,
    learning_rate: float = 5e-2,
    constraint_weight: int = 15
):
    """
    High-level helper: set up feax problem, build J_total(rho), and run MDMM
    topology optimization with a volume fraction constraint.
    """

    # 1) BC preset -> fixed & load locations
    fixed_location, load_location = make_bc_preset(bc_preset_name, Lx, Ly)

    # 2) Compliance functional J_total(rho) from your helper
    J_total = create_J_total(
        mesh=mesh,
        fixed_location=fixed_location,
        load_location=load_location,
        ele_type=ele_type,
        E0=E0,
        E_eps=E_eps,
        nu=nu,
        p=p,
        T=T,
        gauss_order=gauss_order,
        iter_num=iter_num,
        check_convergence=True,
        verbose=False
    )
    rho_shape = mesh.cells.shape[0]

    # 4) Call the generic MDMM driver
    rho_opt, history, final_iter = run_topopt_mdmm(
        J_total=J_total,
        rho_shape=rho_shape,
        vol_frac=vol_frac,
        num_steps=num_steps,
        constraint_weight=constraint_weight,
        print_every=1,
        learning_rate=learning_rate,
    )

    return rho_opt, history, final_iter


# ----------------------------
# Grid search parameters
# ----------------------------
param_grid = {
    "scale": [1],
    "vol_frac": [0.5],
    "learning_rate": [0.1],
    "constraint_weight": [10],
    # "scale": [1],
    # "vol_frac": [0.6],
    # "learning_rate": [0.05],
}

num_steps = 4000
ele_type = "QUAD4"
Lx, Ly = 60.0, 30.0

# Use pathlib Paths
base_dir = Path("outputs")
run_dir = base_dir / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_dir.mkdir(parents=True, exist_ok=True)

results = []
run_id = 0

for cfg in ParameterGrid(param_grid):
    run_id += 1

    scale = cfg["scale"]
    vol_frac = cfg["vol_frac"]
    lr = cfg["learning_rate"]
    constraint_weight = cfg["constraint_weight"]

    Nx = int(Lx * scale)
    Ny = int(Ly * scale)

    mesh = rectangle_mesh(
        Nx, Ny,
        domain_x=Lx,
        domain_y=Ly,
    )

    print(
        f"\nRun {run_id} | "
        f"scale={scale}, "
        f"vol_frac={vol_frac}, "
        f"constraint_weight={constraint_weight}, "
        f"lr={lr}"
    )

    rho_opt, history, final_iter = run_feax_topopt(
        mesh=mesh,
        Lx=Lx,
        Ly=Ly,
        bc_preset_name="cantilever_corner",
        vol_frac=vol_frac,
        num_steps=num_steps,
        learning_rate=lr,
        constraint_weight=constraint_weight
    )

    # ----------------------------
    # Save PNG visualization
    # ----------------------------
    rho_png_path = run_dir / f"rho_scale{scale}_vf{vol_frac}_lr{lr}_cw{constraint_weight}.png"

    save_rho_png(
        rho=rho_opt,
        title=f"scale={scale}, vf={vol_frac}, lr={lr}",
        Nx=Nx,
        Ny=Ny,
        path=rho_png_path,
    )

    results.append(
        {
            **cfg,
            "Nx": Nx,
            "Ny": Ny,
            "Compliance": history["J"][-1],
            "Volume": history["vol"][-1],
            "Iteration": final_iter,
        }
    )

# ----------------------------
# Results table
# ----------------------------
df = pd.DataFrame(results)

df = df.sort_values(
    by=["Compliance"],
    ascending=[True],
).reset_index(drop=True)

print(
    df.to_latex(
        index=False,
        float_format="%.4f",
        caption="Baseline Grid search results for topology optimization",
        label="tab:topopt_grid",
    )
)