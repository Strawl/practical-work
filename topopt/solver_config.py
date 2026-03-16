from __future__ import annotations

from typing import Tuple

from feax import CUDSSMatrixType, CUDSSMatrixView, CUDSSOptions
from feax import DirectSolverOptions, IterativeSolverOptions
from feax.problem import MatrixView


AVAILABLE_SOLVERS: Tuple[str, ...] = (
    "cg",
    "bicgstab",
    "gmres",
    "cudss",
    "spsolve",
    "umfpack",
    "cholmod",
)


def _resolve_name(solver_name: str) -> str:
    solver = solver_name.lower()
    if solver == "lineax":
        return "cg"
    return solver


def _build_one_solver_options(
    solver_name: str,
    *,
    check_convergence: bool,
    verbose: bool,
):
    solver = _resolve_name(solver_name)

    if solver == "cg":
        return IterativeSolverOptions(
            solver="cg",
            tol=1e-5,
            atol=1e-6,
            maxiter=10000,
            use_jacobi_preconditioner=True,
            check_convergence=check_convergence,
            verbose=verbose,
        )
    if solver == "bicgstab":
        return IterativeSolverOptions(
            solver="bicgstab",
            tol=1e-8,
            atol=1e-8,
            maxiter=10000,
            use_jacobi_preconditioner=True,
            check_convergence=check_convergence,
            verbose=verbose,
        )
    if solver == "gmres":
        return IterativeSolverOptions(
            solver="gmres",
            tol=1e-8,
            atol=1e-8,
            maxiter=10000,
            use_jacobi_preconditioner=True,
            check_convergence=check_convergence,
            verbose=verbose,
        )

    cudss_options = CUDSSOptions(
        matrix_type=CUDSSMatrixType.SPD,
        matrix_view=CUDSSMatrixView.UPPER,
        device_id=0,
    )

    if solver == "cudss":
        return DirectSolverOptions(
            solver="cudss",
            cudss_options=cudss_options,
            check_convergence=check_convergence,
            verbose=verbose,
        )
    if solver == "spsolve":
        return DirectSolverOptions(
            solver="spsolve",
            cudss_options=cudss_options,
            check_convergence=check_convergence,
            verbose=verbose,
        )
    if solver == "umfpack":
        return DirectSolverOptions(
            solver="umfpack",
            cudss_options=cudss_options,
            check_convergence=check_convergence,
            verbose=verbose,
        )
    if solver == "cholmod":
        return DirectSolverOptions(
            solver="cholmod",
            cudss_options=cudss_options,
            check_convergence=check_convergence,
            verbose=verbose,
        )

    raise ValueError(
        f"Unknown solver '{solver_name}'. "
        f"Choose from: {sorted(AVAILABLE_SOLVERS)}"
    )


def build_solver_setup(
    fwd_solver_name: str,
    bwd_solver_name: str,
    *,
    check_convergence: bool,
    verbose: bool,
):
    fwd_solver = _resolve_name(fwd_solver_name)
    bwd_solver = _resolve_name(bwd_solver_name)

    fwd_solver_options = _build_one_solver_options(
        fwd_solver,
        check_convergence=check_convergence,
        verbose=verbose,
    )
    bwd_solver_options = _build_one_solver_options(
        bwd_solver,
        check_convergence=check_convergence,
        verbose=verbose,
    )

    upper_view_solvers = {"cholmod", "cudss"}
    if fwd_solver in upper_view_solvers or bwd_solver in upper_view_solvers:
        matrix_view = MatrixView.UPPER
    else:
        matrix_view = MatrixView.FULL

    return fwd_solver_options, bwd_solver_options, matrix_view
