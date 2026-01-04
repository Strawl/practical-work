from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import click

from topopt.jax_setup import init_jax

init_jax()


def _latest_dir(root: Path = Path("./outputs")) -> Path:
    root = root.expanduser().resolve()
    dirs = sorted([p for p in root.glob("*") if p.is_dir()])
    if not dirs:
        raise click.ClickException(f"No output directories found in {root}")
    return dirs[-1]


def _fresh_train_dir(suffix: str) -> Path:
    ts = datetime.now().strftime("%m-%d_%H-%M-%S")
    return (Path("./outputs") / f"{ts}_{suffix}").expanduser().resolve()


@click.group()
def main() -> None:
    """Topopt CLI."""
    pass


@main.command()
@click.option(
    "--config",
    "train_config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--save-dir", type=click.Path(file_okay=False, path_type=Path), default=None
)
def train(train_config_path: Path, save_dir: Optional[Path]) -> None:
    save_dir = (
        (save_dir or _fresh_train_dir(suffix="neural_field")).expanduser().resolve()
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    click.echo(f"Saving data to: {save_dir}")
    from topopt.train import train_from_config

    train_from_config(train_config_path=str(train_config_path), save_dir=save_dir)


@main.command()
@click.option("--scale", type=int, default=None)
@click.option(
    "--save-dir", type=click.Path(file_okay=False, path_type=Path), default=None
)
@click.option(
    "--visualize/--no-visualize",
    default=False,
    help="Show density field visualizations",
)
def evaluate(
    scale: Optional[int],
    save_dir: Optional[Path],
    visualize: bool,
) -> None:
    """
    Evaluate trained models and optionally visualize results.
    """
    base_dir = save_dir.expanduser().resolve() if save_dir else _latest_dir()
    click.echo(f"Using directory (SAVE_DIR): {base_dir}")

    from topopt.evaluation import evaluate_models

    df = evaluate_models(
        save_dir=base_dir,
        scale=scale,
        visualize=visualize,
    )

    click.echo(f"\nEvaluated {len(df)} models.")


@main.command(help="Run FEAX TopOpt MMA baseline and save outputs.")
@click.option("--lx", type=float, default=60.0, show_default=True)
@click.option("--ly", type=float, default=30.0, show_default=True)
@click.option(
    "--save-dir", type=click.Path(file_okay=False, path_type=Path), default=None
)
@click.option("--scale", type=float, default=1.0, show_default=True)
@click.option(
    "--bc-preset",
    "bc_preset_name",
    type=str,
    default="cantilever_corner",
    show_default=True,
)
@click.option("--vol-frac", type=float, default=0.5, show_default=True)
@click.option("--radius", type=float, default=1.0, show_default=True)
@click.option("--max-iter", type=int, default=100, show_default=True)
@click.option("--save-every", type=int, default=10, show_default=True)
@click.option("--print-every", type=int, default=10, show_default=True)
def mma(
    lx: float,
    ly: float,
    save_dir: Optional[Path],
    scale: float,
    bc_preset_name: str,
    vol_frac: float,
    radius: float,
    max_iter: int,
    save_every: int,
    print_every: int,
) -> None:
    save_dir = (save_dir or _fresh_train_dir(suffix="baseline")).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    click.echo(f"Saving MMA results to: {save_dir}")

    from topopt.baseline import run_feax_topopt_mma

    run_feax_topopt_mma(
        Lx=lx,
        Ly=ly,
        save_dir=save_dir,
        scale=scale,
        bc_preset_name=bc_preset_name,
        vol_frac=vol_frac,
        radius=radius,
        max_iter=max_iter,
        save_every=save_every,
        print_every=print_every,
    )
