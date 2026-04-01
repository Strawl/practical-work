from __future__ import annotations

from typing import Callable, Sequence

import jax.numpy as jnp
from feax import DirichletBCSpec


DIRICHLET_BOUNDARY_CONDITION_ALIASES = {
    "cantilever": "cantilever_left_support",
    "left": "cantilever_left_support",
    "left_support": "cantilever_left_support",
    "cantilever_left_clamped": "cantilever_left_support",
}

NEUMANN_BOUNDARY_CONDITION_ALIASES = {
    "corner": "cantilever_corner",
    "mid": "cantilever_mid",
    "middle": "cantilever_mid",
    "top_corner": "cantilever_top_corner",
    "top": "cantilever_top_corner",
    "upper_mid": "cantilever_upper_mid",
    "lower_mid": "cantilever_lower_mid",
    "full": "cantilever_full",
    "continuous": "cantilever_full",
    "double": "cantilever_double",
    "double_spot": "cantilever_double",
    "triple": "cantilever_triple",
    "endcaps": "cantilever_endcaps",
}


def canonicalize_dirichlet_boundary_conditions(name: str) -> str:
    canonical_name = DIRICHLET_BOUNDARY_CONDITION_ALIASES.get(name, name)
    if canonical_name != "cantilever_left_support":
        raise ValueError(f"Unknown Dirichlet boundary conditions preset: {name}")
    return canonical_name


def canonicalize_neumann_boundary_conditions(name: str) -> str:
    canonical_name = NEUMANN_BOUNDARY_CONDITION_ALIASES.get(name, name)
    if canonical_name not in {
        "cantilever_corner",
        "cantilever_mid",
        "cantilever_top_corner",
        "cantilever_upper_mid",
        "cantilever_lower_mid",
        "cantilever_full",
        "cantilever_double",
        "cantilever_triple",
        "cantilever_endcaps",
    }:
        raise ValueError(f"Unknown Neumann boundary conditions preset: {name}")
    return canonical_name


def _cantilever_left_support_location(Lx: float) -> Callable:
    x_tol = 1e-5 * max(1.0, Lx)

    def location_fn(point):
        x, _ = point
        return jnp.isclose(x, 0.0, atol=x_tol)

    return location_fn


def make_dirichlet_boundary_conditions(
    name: str,
    Lx: float,
    Ly: float,
) -> tuple[DirichletBCSpec, ...]:
    del Ly
    name = canonicalize_dirichlet_boundary_conditions(name)

    if name == "cantilever_left_support":
        return (
            DirichletBCSpec(
                location=_cantilever_left_support_location(Lx),
                component="all",
                value=0.0,
            ),
        )

    raise ValueError(f"Unknown Dirichlet boundary conditions preset: {name}")


def make_neumann_boundary_location(Lx: float, Ly: float) -> Callable:
    del Ly
    x_tol = 1e-5 * max(1.0, Lx)

    def location_fn(point):
        x, _ = point
        return jnp.isclose(x, Lx, atol=x_tol)

    return location_fn


def make_neumann_surface_var_fn(
    name: str,
    Lx: float,
    Ly: float,
    traction_value: float,
) -> Callable:
    del Lx
    name = canonicalize_neumann_boundary_conditions(name)
    y_tol = 1e-5 * max(1.0, Ly)

    def _within_band(y, center, half_width):
        return jnp.abs(y - center) <= half_width

    def _spot_mask(y, center, relative_half_width, absolute_min=0.5):
        half_width = max(relative_half_width * Ly + y_tol, absolute_min)
        return _within_band(y, center, half_width)

    if name == "cantilever_corner":
        def surface_var_fn(point):
            _, y = point
            return jnp.where(_spot_mask(y, 0.0, 0.1), traction_value, 0.0)

        return surface_var_fn

    if name == "cantilever_mid":
        def surface_var_fn(point):
            _, y = point
            return jnp.where(_spot_mask(y, 0.5 * Ly, 0.05), traction_value, 0.0)

        return surface_var_fn

    if name == "cantilever_top_corner":
        def surface_var_fn(point):
            _, y = point
            return jnp.where(_spot_mask(y, Ly, 0.1), traction_value, 0.0)

        return surface_var_fn

    if name == "cantilever_upper_mid":
        def surface_var_fn(point):
            _, y = point
            return jnp.where(_spot_mask(y, 0.72 * Ly, 0.06), traction_value, 0.0)

        return surface_var_fn

    if name == "cantilever_lower_mid":
        def surface_var_fn(point):
            _, y = point
            return jnp.where(_spot_mask(y, 0.28 * Ly, 0.06), traction_value, 0.0)

        return surface_var_fn

    if name == "cantilever_full":
        def surface_var_fn(point):
            del point
            return traction_value

        return surface_var_fn

    if name == "cantilever_double":
        def surface_var_fn(point):
            _, y = point
            lower = _spot_mask(y, 0.22 * Ly, 0.05)
            upper = _spot_mask(y, 0.78 * Ly, 0.05)
            return jnp.where(jnp.logical_or(lower, upper), traction_value, 0.0)

        return surface_var_fn

    if name == "cantilever_triple":
        def surface_var_fn(point):
            _, y = point
            lower = _spot_mask(y, 0.18 * Ly, 0.04)
            middle = _spot_mask(y, 0.5 * Ly, 0.04)
            upper = _spot_mask(y, 0.82 * Ly, 0.04)
            is_loaded = jnp.logical_or(lower, jnp.logical_or(middle, upper))
            return jnp.where(is_loaded, traction_value, 0.0)

        return surface_var_fn

    if name == "cantilever_endcaps":
        def surface_var_fn(point):
            _, y = point
            bottom = _spot_mask(y, 0.0, 0.08)
            top = _spot_mask(y, Ly, 0.08)
            return jnp.where(jnp.logical_or(bottom, top), traction_value, 0.0)

        return surface_var_fn

    raise ValueError(f"Unknown Neumann boundary conditions preset: {name}")


def make_neumann_surface_var_fns(
    names: Sequence[str],
    Lx: float,
    Ly: float,
    traction_value: float,
) -> tuple[tuple[str, Callable], ...]:
    return tuple(
        (
            canonicalize_neumann_boundary_conditions(name),
            make_neumann_surface_var_fn(name, Lx, Ly, traction_value),
        )
        for name in names
    )
