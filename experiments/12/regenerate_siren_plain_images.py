from __future__ import annotations

import argparse
from pathlib import Path

import feax.gene as gene
import jax.numpy as jnp
from feax.gene import create_helmholtz_filter
from feax.mesh import rectangle_mesh
from jax.nn import sigmoid
import numpy as np
from PIL import Image

from topopt.fem_utils import get_element_geometry
from topopt.jax_setup import init_jax
from topopt.serialization import TrainingConfig, load_model_from_config


def save_plain_density_png(rho, nx: int, ny: int, path: Path) -> None:
    rho_img = jnp.reshape(jnp.asarray(rho), (ny, nx), order="F")
    rho_np = np.asarray(rho_img, dtype=np.float32)
    rho_np = np.flipud(rho_np)
    rho_np = np.clip(rho_np, 0.0, 1.0)
    grayscale = np.rint((1.0 - rho_np) * 255.0).astype(np.uint8)
    image = Image.fromarray(grayscale, mode="L").resize((3000, 1500), Image.Resampling.NEAREST)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def generate_plain_image(
    *,
    run_dir: Path,
    model_name: str,
    scale: int,
    output_path: Path,
    heaviside_beta: float,
    heaviside_threshold: float,
) -> None:
    train_config = TrainingConfig.from_yaml(run_dir / "training_config_snapshot.yaml")
    lx = float(train_config.training.Lx)
    ly = float(train_config.training.Ly)
    nx = int(lx * scale)
    ny = int(ly * scale)

    mesh = rectangle_mesh(Nx=nx, Ny=ny, domain_x=lx, domain_y=ly)
    coords = get_element_geometry(mesh)["centroids_scaled"]

    cfg_path = run_dir / f"{model_name}_config.yaml"
    model, _, _, _ = load_model_from_config(cfg_path, run_dir)

    rho_raw = sigmoid(model(coords))
    radius = float(train_config.training.helmholtz_radius)
    if radius > 0.0:
        rho_filtered = create_helmholtz_filter(mesh, radius)(rho_raw)
    else:
        rho_filtered = rho_raw

    if heaviside_beta > 0.0:
        rho_eval = gene.heaviside_projection(
            rho_filtered,
            beta=heaviside_beta,
            threshold=heaviside_threshold,
        )
    else:
        rho_eval = rho_filtered

    save_plain_density_png(rho_eval, nx, ny, output_path)
    print(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate plain SIREN density PNGs without metadata overlays."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", default="model_3")
    parser.add_argument("--heaviside-beta", type=float, default=10.0)
    parser.add_argument("--heaviside-threshold", type=float, default=0.5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/12"),
    )
    return parser.parse_args()


def main() -> None:
    init_jax()
    args = parse_args()

    generate_plain_image(
        run_dir=args.run_dir,
        model_name=args.model,
        scale=3,
        output_path=args.output_dir / "siren_model_4_scale3.png",
        heaviside_beta=args.heaviside_beta,
        heaviside_threshold=args.heaviside_threshold,
    )
    generate_plain_image(
        run_dir=args.run_dir,
        model_name=args.model,
        scale=15,
        output_path=args.output_dir / "siren_model_4_scale15.png",
        heaviside_beta=args.heaviside_beta,
        heaviside_threshold=args.heaviside_threshold,
    )


if __name__ == "__main__":
    main()
