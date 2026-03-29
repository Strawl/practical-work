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
}


def canonicalize_dirichlet_boundary_conditions(name: str) -> str:
    canonical_name = DIRICHLET_BOUNDARY_CONDITION_ALIASES.get(name, name)
    if canonical_name != "cantilever_left_support":
        raise ValueError(f"Unknown Dirichlet boundary conditions preset: {name}")
    return canonical_name


def canonicalize_neumann_boundary_conditions(name: str) -> str:
    canonical_name = NEUMANN_BOUNDARY_CONDITION_ALIASES.get(name, name)
    if canonical_name not in {"cantilever_corner", "cantilever_mid"}:
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

    if name == "cantilever_corner":
        corner_band = max(0.1 * Ly + y_tol, 0.5)

        def surface_var_fn(point):
            _, y = point
            return jnp.where(y <= corner_band, traction_value, 0.0)

        return surface_var_fn

    if name == "cantilever_mid":
        mid_band = max(0.05 * Ly + y_tol, 0.5)

        def surface_var_fn(point):
            _, y = point
            return jnp.where(jnp.abs(y - 0.5 * Ly) <= mid_band, traction_value, 0.0)

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
