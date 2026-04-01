from pathlib import Path
from typing import Callable, Optional

import equinox as eqx
import jax.numpy as jnp
import optax
from feax.mesh import Mesh, rectangle_mesh
from tqdm import tqdm

import jax
from topopt.bc import (
    make_dirichlet_boundary_conditions,
    make_neumann_boundary_location,
    make_neumann_surface_var_fns,
)
from topopt.fem_utils import (
    create_surface_vars,
    create_objective_functions,
    get_element_geometry,
)
from topopt.monitoring import MetricTracker, StepTimer
from topopt.serialization import (
    TrainingConfig,
    TrainingHyperparams,
    create_models,
    serialize_ensemble,
)





def batched_loss_and_grad(
    models,
    target_densities,
    compliance_normalizers,
    lams,
    penalties,
    coords,
    surface_vars,
    compliance_fn,
    volume_fraction_fn,
):
    def batched_loss(
        models,
        target_densities,
        compliance_normalizers,
        lams,
        penalties,
        coords,
        surface_vars,
        compliance_fn,
        volume_fraction_fn,
    ):
        rhos = jax.vmap(lambda m: jax.nn.sigmoid(m(coords)))(models)
        true_compliances = compliance_fn(rhos, surface_vars)
        normalized_compliances = (
            true_compliances / compliance_normalizers
        )

        vol_frac = volume_fraction_fn(rhos)
        vol_frac_error = vol_frac - target_densities
        normalized_constraint = (
            vol_frac_error * 10
        )
        normalized_violation = jnp.maximum(normalized_constraint, 0.0)

        al_linear = lams * normalized_violation
        al_quadratic = 0.5 * penalties * normalized_violation**2
        al_term = al_linear + al_quadratic

        losses = normalized_compliances + al_term
        aux = (
            normalized_constraint,
            normalized_violation,
            true_compliances,
            normalized_compliances,
            vol_frac,
            vol_frac_error,
            al_linear,
            al_quadratic,
            al_term,
        )
        return jnp.sum(losses), (losses, aux)

    batched_loss_and_grad_fn = eqx.filter_value_and_grad(batched_loss, has_aux=True)
    (total_loss, (losses, aux)), grads = batched_loss_and_grad_fn(
        models,
        target_densities,
        compliance_normalizers,
        lams,
        penalties,
        coords,
        surface_vars,
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
    compliance_normalizers,
    lams,
    penalties,
    coords,
    surface_vars,
    compliance_fn,
    volume_fraction_fn,
):
    losses, aux, grads = batched_loss_and_grad(
        models,
        target_densities,
        compliance_normalizers,
        lams,
        penalties,
        coords,
        surface_vars,
        compliance_fn,
        volume_fraction_fn,
    )

    updates, new_opt_states = jax.vmap(optimizer.update)(
        grads, opt_states, value=losses
    )
    new_models = jax.vmap(eqx.apply_updates)(models, updates)

    return new_models, new_opt_states, losses, aux


def train_model_batch(
    models,
    mesh: Mesh,
    target_densities: jnp.ndarray,
    penalties: jnp.ndarray,
    compliance_normalizers: jnp.ndarray,
    surface_vars,
    compliance_fn: Callable[[jnp.ndarray, object], float],
    volume_fraction_fn: Callable[[jnp.ndarray], float],
    hyperparameters: TrainingHyperparams,
    tracker=MetricTracker,
):
    lr = hyperparameters.lr
    lr_schedule_config = hyperparameters.learning_rate_schedule

    step_timer = StepTimer()
    wal_timer = StepTimer()
    wal_timer.start()

    geom = get_element_geometry(mesh)
    coords = centroids_scaled = geom["centroids_scaled"]
    num_elements = geom["num_elements"]
    dx = geom["dx_scaled"]
    dy = geom["dy_scaled"]

    compliance_fn = jax.vmap(compliance_fn, in_axes=(0, 0))
    volume_fraction_fn = jax.vmap(volume_fraction_fn)

    if lr_schedule_config is None:
        lr_scale_schedule = optax.polynomial_schedule(
            init_value=1.0,
            end_value=0.1,
            power=0.9,
            transition_steps=80,
            transition_begin=20,
        )
    else:
        lr_scale_schedule = optax.polynomial_schedule(
            init_value=lr_schedule_config.init_value,
            end_value=lr_schedule_config.end_value,
            power=lr_schedule_config.power,
            transition_steps=lr_schedule_config.transition_steps,
            transition_begin=lr_schedule_config.transition_begin,
        )

    optimizer = optax.chain(
        optax.zero_nans(),
        optax.clip_by_global_norm(hyperparameters.grad_clip_norm),
        optax.adabelief(learning_rate=lambda count: lr * lr_scale_schedule(count)),
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
            compliance_normalizers,
            lams,
            penalties,
            coords,
            surface_vars,
            compliance_fn=compliance_fn,
            volume_fraction_fn=volume_fraction_fn,
        )
        step_time_s = step_timer.stop(block_on=jnp.mean(losses))
        if iteration > 1:
            tracker.log("optimisation_step_wall_time_s", step_time_s)
        else:
            compile_time += step_time_s

        (
            normalized_constraints,
            normalized_violations,
            true_compliances,
            normalized_compliances,
            volume_fractions,
            vol_frac_errors,
            al_linears,
            al_quadratics,
            al_terms,
        ) = aux

        good = jnp.isfinite(normalized_constraints)
        updated_lams = lams + penalties * normalized_constraints
        lams = jnp.where(good, updated_lams, lams)
        lam_updates = lams - old_lams

        # Monitoring
        lr_scale = float(lr_scale_schedule(iteration))
        lr_scales = jnp.full(losses.shape, lr_scale, dtype=float)
        effective_lrs = lr * lr_scales
        tracker.log("loss", losses)
        tracker.log("true_compliance", true_compliances)
        tracker.log("normalized_compliance", normalized_compliances)
        tracker.log("volume_fraction", volume_fractions)
        tracker.log("constraint_value", normalized_constraints)
        tracker.log("constraint_violation", normalized_violations)
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
        aux_items = (
            ("normalized_constraint", normalized_constraints),
            ("normalized_violation", normalized_violations),
            ("true_compliance", true_compliances),
            ("normalized_compliance", normalized_compliances),
            ("volume_fraction", volume_fractions),
            ("volume_fraction_error", vol_frac_errors),
            ("al_linear", al_linears),
            ("al_quadratic", al_quadratics),
            ("al_term", al_terms),
        )
        for name, values in aux_items:
            values_flat = jnp.ravel(jnp.asarray(values))
            values_str = " | ".join(
                [f"{float(values_flat[i]):.6f}" for i in range(len(values_flat))]
            )
            print(f"  {name}: [{values_str}]")

    wal_time = wal_timer.stop()

    if tracker.steps("optimisation_step_wall_time_s") > 0:
        opt_hist = tracker.stack("optimisation_step_wall_time_s")
        hot_time = float(jnp.sum(opt_hist))
    else:
        hot_time = 0.0

    return models, opt_states, hot_time, wal_time, compile_time


def train_from_config(
    train_config_path: Path,
    save_dir: Path,
    *,
    fwd_linear_solver: str = "spsolve",
    bwd_linear_solver: Optional[str] = None,
):
    train_config: TrainingConfig = TrainingConfig.from_yaml(train_config_path)
    if bwd_linear_solver is None:
        bwd_linear_solver = fwd_linear_solver

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    tracker = MetricTracker(save_dir=save_dir)
    ele_type = "QUAD4"

    Lx = train_config.training.Lx
    Ly = train_config.training.Ly
    scale = train_config.training.scale
    Nx = int(Lx * scale)
    Ny = int(Ly * scale)
    mesh = rectangle_mesh(Nx, Ny, domain_x=Lx, domain_y=Ly)

    dirichlet_boundary_conditions = make_dirichlet_boundary_conditions(
        train_config.training.dirichlet_boundary_conditions,
        Lx,
        Ly,
    )
    model_neumann_boundary_conditions = [
        model.training.neumann_boundary_conditions for model in train_config.models
    ]
    unique_neumann_boundary_conditions = tuple(
        dict.fromkeys(model_neumann_boundary_conditions)
    )
    neumann_boundary_location = make_neumann_boundary_location(Lx, Ly)
    named_surface_var_fns = make_neumann_surface_var_fns(
        unique_neumann_boundary_conditions,
        Lx,
        Ly,
        traction_value=1e2,
    )

    solve_forward, evaluate_volume, filter_fn, _, _, problem = create_objective_functions(
        mesh,
        dirichlet_boundary_conditions,
        neumann_boundary_location,
        ele_type=ele_type,
        check_convergence=True,
        verbose=True,
        radius=train_config.training.helmholtz_radius,
        fwd_linear_solver=fwd_linear_solver,
        bwd_linear_solver=bwd_linear_solver,
    )

    rng = jax.random.PRNGKey(train_config.training.model_rng_seed)
    model_batch, target_densities, penalties = create_models(
        train_config,
        rng,
    )
    surface_vars_by_neumann_boundary_conditions = dict(
        zip(
            unique_neumann_boundary_conditions,
            create_surface_vars(
                problem,
                tuple(surface_var_fn for _, surface_var_fn in named_surface_var_fns),
            ),
        )
    )
    model_surface_vars = [
        surface_vars_by_neumann_boundary_conditions[
            model.training.neumann_boundary_conditions
        ]
        for model in train_config.models
    ]
    neumann_boundary_condition_counts = {
        neumann_boundary_condition: model_neumann_boundary_conditions.count(
            neumann_boundary_condition
        )
        for neumann_boundary_condition in unique_neumann_boundary_conditions
    }
    print(
        "Using Neumann boundary conditions: "
        + ", ".join(
            f"{neumann_boundary_condition} "
            f"({neumann_boundary_condition_counts[neumann_boundary_condition]} models)"
            for neumann_boundary_condition in unique_neumann_boundary_conditions
        )
    )
    surface_vars = jax.tree_util.tree_map(
        lambda *xs: jnp.stack(xs, axis=0),
        *model_surface_vars,
    )

    full_material_rho = jnp.ones(problem.num_cells)
    no_material_rho = jnp.zeros(problem.num_cells)
    full_material_compliances = []
    print("Compliance bounds after solver warm-up")
    for model_index, (neumann_boundary_condition, model_surface_var) in enumerate(
        zip(model_neumann_boundary_conditions, model_surface_vars),
        start=1,
    ):
        full_material_compliance = float(
            solve_forward(full_material_rho, model_surface_var)
        )
        void_compliance = float(solve_forward(no_material_rho, model_surface_var))
        full_material_compliances.append(full_material_compliance)
        print(
            f"  model {model_index:02d} [{neumann_boundary_condition}] "
            f"full = {full_material_compliance:.6f} | "
            f"void = {void_compliance:.6f}"
        )
    compliance_normalizers = jnp.asarray(full_material_compliances, dtype=float)

    trained_models, opt_states, hot_time, wal_time, compile_time = train_model_batch(
        models=model_batch,
        mesh=mesh,
        target_densities=target_densities,
        penalties=penalties,
        compliance_normalizers=compliance_normalizers,
        surface_vars=surface_vars,
        compliance_fn=solve_forward,
        volume_fraction_fn=evaluate_volume,
        hyperparameters=train_config.training,
        tracker=tracker,
    )

    other_time = wal_time - hot_time - compile_time
    share_hot = hot_time / wal_time
    share_compile = compile_time / wal_time
    share_other = other_time / wal_time

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

    serialize_ensemble(trained_models, opt_states, train_config, save_dir=save_dir)
