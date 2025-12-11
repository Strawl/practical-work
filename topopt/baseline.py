from typing import Callable
import jax
import jax.numpy as jnp
import optax
from feax.experimental.topopt_toolkit import mdmm
from feax.mesh import rectangle_mesh, Mesh

from bc import make_bc_preset
from fem_utils import create_J_total
import matplotlib.pyplot as plt




def run_topopt_mdmm(
    J_total: Callable[[jnp.ndarray], jnp.ndarray],
    rho_shape,
    vol_frac: float = 0.4,
    num_steps: int = 1000,
    learning_rate: float = 5e-2,
    constraint_damping: float = 1.0,
    constraint_weight: float = 1.0,
    print_every: int = 10,
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

    volume_constraint = mdmm.eq(
        lambda rho: vol_frac - jnp.mean(rho),
        damping=constraint_damping,
        weight=constraint_weight,
        reduction=lambda x: x,
    )

    initial_rho = jnp.full(rho_shape, vol_frac)
    alpha0 = jnp.log(initial_rho / (1.0 - initial_rho))

    mdmm_params0 = volume_constraint.init(initial_rho)
    print(mdmm_params0)

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
        total_loss = J + c_loss

        aux = {
            "J": J,
            "vol": jnp.mean(rho),
            "infeas": infeas,
            "rho": rho,
        }
        return total_loss, aux

    optimizer = optax.chain(
        optax.adabelief(learning_rate),
        mdmm.optax_prepare_update(),
    )
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state):
        (loss_val, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val, aux

    history = {"J": [], "vol": [], "infeas": [], "rho": []}

    for it in range(num_steps):
        params, opt_state, loss_val, aux = step(params, opt_state)

        J_val = float(aux["J"])
        vol_val = float(aux["vol"])
        infeas_val = float(aux["infeas"])

        history["J"].append(J_val)
        history["vol"].append(vol_val)
        history["infeas"].append(infeas_val)
        history["rho"].append(aux["rho"])

        if (it % print_every) == 0 or it == num_steps - 1:
            print(
                f"[{it:05d}] "
                f"J = {J_val}, "
                f"vol = {vol_val}, "
                f"infeas = {infeas_val}"
            )

    # Final density field
    rho_opt = jax.nn.sigmoid(params["alpha"])
    return rho_opt, history   

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
    )
    rho_shape = mesh.cells.shape[0]

    # 4) Call the generic MDMM driver
    rho_opt, history = run_topopt_mdmm(
        J_total=J_total,
        rho_shape=rho_shape,
        vol_frac=vol_frac,
        num_steps=num_steps,
        constraint_weight=15,
        learning_rate=learning_rate,
    )

    return rho_opt, history

ele_type = "QUAD4"
Lx, Ly = 60.0, 30.0
scale = 1
Nx = 60 * scale
Ny = 30 * scale
mesh = rectangle_mesh(Nx, Ny, domain_x=Lx, domain_y=Ly)

rho_opt, history = run_feax_topopt(
    mesh=mesh,
    Lx=Lx,
    Ly=Ly,
    bc_preset_name="cantilever_corner",
    vol_frac=0.5,
    num_steps=500,
    learning_rate=0.05,
)



def plot_rho_evolution(rho_list, Nx, Ny, interval=50):
    """
    Plot the evolution of density fields during topology optimization.

    Args:
        rho_list: list of 1D arrays (jnp.ndarray or np.ndarray) of length Nx * Ny
            Densities at different iterations.
        Nx, Ny : int
            Number of elements in x and y directions.
        interval: int
            Plot every `interval` steps (to reduce frames for long runs).
    """
    n = len(rho_list)
    steps = list(range(0, n, interval))
    n_plots = len(steps)

    plt.figure(figsize=(3 * n_plots, 3))
    for i, k in enumerate(steps):
        rho_flat = jnp.asarray(rho_list[k])
        rho = jnp.reshape(rho_flat, (Ny, Nx), order="F")

        plt.subplot(1, n_plots, i + 1)
        plt.imshow(rho, cmap="gray_r", origin="lower", vmin=0, vmax=1)
        plt.title(f"iter {k}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

plot_rho_evolution(history["rho"], Nx=Nx, Ny=Ny)