import sys
from typing import Callable

import config
import equinox as eqx
import jax.numpy as jnp
import optax
from bc import make_bc_preset
from feax.mesh import Mesh, rectangle_mesh
from fem_utils import create_objective_functions, get_element_geometry
from monitoring import MetricTracker, StepTimer
from serialization import (
    TrainingConfig,
    TrainingHyperparams,
    create_models,
    serialize_ensemble,
)
from tqdm import tqdm

import jax


def loss(
    model,
    target_vol_frac,
    lambda_,
    mu,
    coords,
    compliance_fn,
    volume_fraction_fn,
):
    rho = jax.nn.sigmoid(model(coords))
    compliance = compliance_fn(rho)

    vol_frac = volume_fraction_fn(rho)
    vol_frac_error = vol_frac - target_vol_frac
    violation = jnp.maximum(vol_frac_error, 0.0)

    al_linear = lambda_ * violation
    al_quadratic = mu * violation**2
    al_term = al_linear + al_quadratic

    total_loss = compliance + al_term

    jax.debug.print(
        "vol_frac={vf:.4f} | violation={v:.4f} | comp={c:.4f} | al={al:.4f} | loss={L:.4f}",
        vf=vol_frac,
        v=violation,
        c=compliance,
        al=al_term,
        L=total_loss,
    )
    aux = (violation, compliance, vol_frac_error, al_linear, al_quadratic, al_term)
    return total_loss, aux


loss_and_grad = eqx.filter_value_and_grad(loss, has_aux=True)


def batched_loss_and_grad(
    models,
    target_densities,
    lams,
    penalties,
    coords,
    compliance_fn,
    volume_fraction_fn,
):
    (losses, aux), grads = jax.vmap(
        loss_and_grad,
        in_axes=(0, 0, 0, 0, None, None, None),
    )(
        models,
        target_densities,
        lams,
        penalties,
        coords,
        compliance_fn,
        volume_fraction_fn,
    )
    return losses, aux, grads


batched_loss_and_grad = jax.jit(
    batched_loss_and_grad,
    static_argnames=(
        "compliance_fn",
        "volume_fraction_fn",
    ),
)


def optimisation_step(
    models,
    opt_states,
    optimizer,
    target_densities,
    lams,
    penalties,
    coords,
    compliance_fn,
    volume_fraction_fn,
):
    losses, aux, grads = batched_loss_and_grad(
        models,
        target_densities,
        lams,
        penalties,
        coords,
        compliance_fn,
        volume_fraction_fn,
    )

    updates, new_opt_states = jax.vmap(optimizer.update)(
        grads, opt_states, value=losses
    )
    new_models = jax.vmap(eqx.apply_updates)(models, updates)

    return new_models, new_opt_states, losses, aux


def train_multiple_models(
    models,
    mesh: Mesh,
    target_densities: jnp.ndarray,
    penalties: jnp.ndarray,
    compliance_fn: Callable[[jnp.ndarray], float],
    volume_fraction_fn: Callable[[jnp.ndarray], float],
    hyperparameters: TrainingHyperparams,
):
    platue_config = hyperparameters.plateau
    lr = hyperparameters.lr

    tracker = MetricTracker(output_dir=config.SAVE_DIR)
    step_timer = StepTimer()
    wal_timer = StepTimer()
    wal_timer.start()

    geom = get_element_geometry(mesh)
    coords = centroids_scaled = geom["centroids_scaled"]
    num_elements = geom["num_elements"]
    dx = geom["dx_scaled"]
    dy = geom["dy_scaled"]

    inner_optim = optax.chain(
        optax.clip_by_global_norm(hyperparameters.grad_clip_norm),
        optax.adabelief(lr),
        optax.contrib.reduce_on_plateau(
            patience=platue_config.patience,
            cooldown=platue_config.cooldown,
            factor=platue_config.factor,
            rtol=platue_config.rtol,
            accumulation_size=platue_config.accumulation_size,
        ),
    )

    optimizer = optax.apply_if_finite(
        inner_optim,
        max_consecutive_errors=30,
    )
    opt_states = jax.vmap(lambda m: optimizer.init(eqx.filter(m, eqx.is_array)))(models)

    rng = jax.random.PRNGKey(hyperparameters.model_rng_seed)
    lams = jnp.zeros_like(target_densities)
    compile_time = 0

    for iteration in tqdm(range(hyperparameters.num_iterations), desc="Iterations"):
        if hyperparameters.jitted_coords:
            # uniform jitter in [-0.5, 0.5] * element size
            rng, kx, ky = jax.random.split(rng, 3)
            ux = jax.random.uniform(kx, (num_elements,), minval=-0.5, maxval=0.5)
            uy = jax.random.uniform(ky, (num_elements,), minval=-0.5, maxval=0.5)

            coords = jnp.stack(
                [
                    centroids_scaled[:, 0] + ux * dx,
                    centroids_scaled[:, 1] + uy * dy,
                ],
                axis=-1,
            )
        old_lams = lams
        step_timer.start()
        models, opt_states, losses, aux = optimisation_step(
            models,
            opt_states,
            optimizer,
            target_densities,
            lams,
            penalties,
            coords,
            compliance_fn=compliance_fn,
            volume_fraction_fn=volume_fraction_fn,
        )
        step_time_s = step_timer.stop(block_on=jnp.mean(losses))
        if iteration > 3:
            tracker.log("optimisation_step_wall_time_s", step_time_s)
        else:
            compile_time += step_time_s

        (
            violations,
            compliances,
            vol_frac_errors,
            al_linears,
            al_quadratics,
            al_terms,
        ) = aux

        good = jnp.isfinite(violations)
        lams = jnp.where(good, lams + 2.0 * penalties * violations, lams)
        lam_updates = lams - old_lams

        # Monitoring
        lr_scales = jax.vmap(lambda s: optax.tree.get(s, "scale"))(opt_states)
        effective_lrs = lr * lr_scales
        tracker.log("loss", losses)
        tracker.log("compliance", compliances)
        tracker.log("constraint_violation", violations)
        tracker.log("volume_fraction_error", vol_frac_errors)
        tracker.log("al_linear", al_linears)
        tracker.log("al_quadratic", al_quadratics)
        tracker.log("al_penalty", al_terms)
        tracker.log("lambda", lams)
        tracker.log("lambda_update", lam_updates)
        tracker.log("learning_rate_scale", lr_scales)
        tracker.log("effective_lr", effective_lrs)

        loss_values = jnp.array(losses)
        loss_str = " | ".join(
            [f"{loss_values[i]:.6f}" for i in range(len(loss_values))]
        )
        mean_loss = float(jnp.mean(loss_values))
        print(
            f"Iteration {iteration:03d} | mean loss = {mean_loss:.6f} | individual = [{loss_str}]"
        )

    wal_time = wal_timer.stop()

    opt_hist = tracker.stack("optimisation_step_wall_time_s")
    hot_time = float(jnp.sum(opt_hist))

    return models, opt_states, tracker, hot_time, wal_time, compile_time


def main(train_config_path: str = config.TRAIN_CONFIG_PATH):
    # ---------------- FE Problem Setup ----------------

    train_config: TrainingConfig = TrainingConfig.from_yaml(train_config_path)
    ele_type = "QUAD4"

    Lx = train_config.training.Lx
    Ly = train_config.training.Ly
    scale = train_config.training.scale
    Nx = int(Lx * scale)
    Ny = int(Ly * scale)
    mesh = rectangle_mesh(Nx, Ny, domain_x=Lx, domain_y=Ly)

    fixed_location, load_location = make_bc_preset("cantilever_corner", Lx, Ly)

    solve_forward, evaluate_volume, _, _ = create_objective_functions(
        mesh,
        fixed_location,
        load_location,
        ele_type=ele_type,
        check_convergence=True,
        verbose=True,
    )

    # ---------------- MODEL Definition ----------------

    rng = jax.random.PRNGKey(train_config.training.model_rng_seed)
    model_batch, target_densities, penalties = create_models(
        train_config,
        rng,
    )

    trained_models, opt_states, tracker, hot_time, wal_time, compile_time = (
        train_multiple_models(
            models=model_batch,
            mesh=mesh,
            target_densities=target_densities,
            penalties=penalties,
            compliance_fn=solve_forward,
            volume_fraction_fn=evaluate_volume,
            hyperparameters=train_config.training,
        )
    )

    other_time = wal_time - hot_time - compile_time
    share_hot = hot_time / wal_time if wal_time > 0 else float("nan")
    share_compile = compile_time / wal_time if wal_time > 0 else float("nan")
    share_other = other_time / wal_time if wal_time > 0 else float("nan")

    print(
        "Timing summary\n"
        f"  wall total    : {wal_time:8.3f}s (100.00%)\n"
        f"  hot optimise  : {hot_time:8.3f}s ({100.0 * share_hot:6.2f}%)\n"
        f"  compile/first : {compile_time:8.3f}s ({100.0 * share_compile:6.2f}%)\n"
        f"  other         : {other_time:8.3f}s ({100.0 * share_other:6.2f}%)"
    )

    tracker.save()
    model_names = [f"siren_{i}" for i in range(1, len(train_config.models) + 1)]
    tracker.plot_all_metrics_across_models(
        model_names=model_names, save=True, show=False
    )

    serialize_ensemble(trained_models, opt_states, train_config)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
