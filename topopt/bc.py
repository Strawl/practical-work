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
    Ny: int | None = None,
) -> Callable:
    """Create a surface-variation function for Neumann boundary conditions.

    Parameters
    ----------
    name : str
        Neumann boundary condition preset name.
    Lx, Ly : float
        Physical domain dimensions.
    traction_value : float
        Traction magnitude (force per unit area).
    Ny : int | None
        Number of elements in the y-direction. When provided, spot masks are
        sized to cover exactly one element face, matching the equivalent point
        force in the original TOuNN implementation. When None, legacy broad
        spot masks are used (``absolute_min=0.5``).

    Returns
    -------
    Callable
        Function ``surface_var_fn(point) -> float`` evaluating the traction
        magnitude at the given boundary point.
    """
    del Lx
    name = canonicalize_neumann_boundary_conditions(name)
    y_tol = 1e-5 * max(1.0, Ly)

    # Element size in y-direction; used when Ny is provided to size masks
    # so they cover exactly one element face on the right edge.
    element_size = Ly / Ny if Ny is not None and Ny > 0 else None

    def _within_band(y, center, half_width):
        return jnp.abs(y - center) <= half_width

    def _spot_mask(y, center, relative_half_width, absolute_min=0.5):
        """Legacy broad mask (default when Ny is not known)."""
        half_width = max(relative_half_width * Ly + y_tol, absolute_min)
        return _within_band(y, center, half_width)

    def _single_element_mask(y, center):
        """Mask covering exactly one element face (half-width = element_size/2)."""
        assert element_size is not None
        return _within_band(y, center, 0.5 * element_size)

    # Choose mask function based on whether mesh resolution is provided
    mask = _single_element_mask if element_size is not None else _spot_mask

    # For corner presets, center the mask on the first element face when using
    # single-element masks (element center = element_size/2), otherwise use 0.
    def corner_center():
        return 0.5 * element_size if element_size is not None else 0.0

    # For mid preset, center is always Ly/2
    mid_center = 0.5 * Ly

    if name == "cantilever_corner":
        cc = corner_center()

        def surface_var_fn(point):
            _, y = point
            if element_size is not None:
                return jnp.where(mask(y, cc), traction_value, 0.0)
            return jnp.where(_spot_mask(y, 0.0, 0.1), traction_value, 0.0)

        return surface_var_fn

    if name == "cantilever_mid":
        def surface_var_fn(point):
            _, y = point
            if element_size is not None:
                return jnp.where(mask(y, mid_center), traction_value, 0.0)
            return jnp.where(_spot_mask(y, 0.5 * Ly, 0.05), traction_value, 0.0)

        return surface_var_fn

    if name == "cantilever_top_corner":
        tc = Ly - 0.5 * element_size if element_size is not None else Ly

        def surface_var_fn(point):
            _, y = point
            if element_size is not None:
                return jnp.where(mask(y, tc), traction_value, 0.0)
            return jnp.where(_spot_mask(y, Ly, 0.1), traction_value, 0.0)

        return surface_var_fn

    if name == "cantilever_upper_mid":
        um = 0.72 * Ly

        def surface_var_fn(point):
            _, y = point
            if element_size is not None:
                return jnp.where(mask(y, um), traction_value, 0.0)
            return jnp.where(_spot_mask(y, 0.72 * Ly, 0.06), traction_value, 0.0)

        return surface_var_fn

    if name == "cantilever_lower_mid":
        lm = 0.28 * Ly

        def surface_var_fn(point):
            _, y = point
            if element_size is not None:
                return jnp.where(mask(y, lm), traction_value, 0.0)
            return jnp.where(_spot_mask(y, 0.28 * Ly, 0.06), traction_value, 0.0)

        return surface_var_fn

    if name == "cantilever_full":
        def surface_var_fn(point):
            del point
            return traction_value

        return surface_var_fn

    if name == "cantilever_double":
        dl = 0.22 * Ly
        du = 0.78 * Ly

        def surface_var_fn(point):
            _, y = point
            if element_size is not None:
                lower = mask(y, dl)
                upper = mask(y, du)
                return jnp.where(jnp.logical_or(lower, upper), traction_value, 0.0)
            lower = _spot_mask(y, 0.22 * Ly, 0.05)
            upper = _spot_mask(y, 0.78 * Ly, 0.05)
            return jnp.where(jnp.logical_or(lower, upper), traction_value, 0.0)

        return surface_var_fn

    if name == "cantilever_triple":
        tl = 0.18 * Ly
        tm = 0.5 * Ly
        tu = 0.82 * Ly

        def surface_var_fn(point):
            _, y = point
            if element_size is not None:
                lower = mask(y, tl)
                middle = mask(y, tm)
                upper = mask(y, tu)
                is_loaded = jnp.logical_or(lower, jnp.logical_or(middle, upper))
                return jnp.where(is_loaded, traction_value, 0.0)
            lower = _spot_mask(y, 0.18 * Ly, 0.04)
            middle = _spot_mask(y, 0.5 * Ly, 0.04)
            upper = _spot_mask(y, 0.82 * Ly, 0.04)
            is_loaded = jnp.logical_or(lower, jnp.logical_or(middle, upper))
            return jnp.where(is_loaded, traction_value, 0.0)

        return surface_var_fn

    if name == "cantilever_endcaps":
        eb = 0.5 * element_size if element_size is not None else 0.0
        et = Ly - 0.5 * element_size if element_size is not None else Ly

        def surface_var_fn(point):
            _, y = point
            if element_size is not None:
                bottom = mask(y, eb)
                top = mask(y, et)
                return jnp.where(jnp.logical_or(bottom, top), traction_value, 0.0)
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
    Ny: int | None = None,
) -> tuple[tuple[str, Callable], ...]:
    return tuple(
        (
            canonicalize_neumann_boundary_conditions(name),
            make_neumann_surface_var_fn(name, Lx, Ly, traction_value, Ny=Ny),
        )
        for name in names
    )


def equivalent_traction_for_point_load(
    name: str,
    *,
    point_load_magnitude: float,
    Ly: float,
    Ny: int,
) -> float:
    """Map a target point-load magnitude to traction for a BC load patch."""
    canonical_name = canonicalize_neumann_boundary_conditions(name)
    if Ny <= 0 or Ly <= 0.0:
        raise ValueError(f"Invalid dimensions: Ly={Ly}, Ny={Ny}")

    edge_segment_length = Ly / Ny
    loaded_segment_count = {
        "cantilever_corner": 1,
        "cantilever_mid": 1,
        "cantilever_top_corner": 1,
        "cantilever_upper_mid": 1,
        "cantilever_lower_mid": 1,
        "cantilever_double": 2,
        "cantilever_triple": 3,
        "cantilever_endcaps": 2,
        "cantilever_full": Ny,
    }[canonical_name]
    return point_load_magnitude / (loaded_segment_count * edge_segment_length)
