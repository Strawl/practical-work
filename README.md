# Topology Optimization with FEAX

This project performs topology optimization using **FEAX**, a finite element analysis library written in JAX.  
Follow the instructions below to set up your environment and run training, evaluation, and baseline optimization.

## Installation

### Ubuntu / Debian

Install topopt:

```bash
sudo apt-get install libatlas-base-dev libblas-dev liblapack-dev libhdf5-dev libglu1-mesa libxi-dev libxmu-dev libglu1-mesa-dev libxinerama1
uv sync --group cuda12
uv run pip install --no-build-isolation --config-settings=cmake.args="-DBUILD_PBATCH_SOLVE=OFF" "spineax[cuda12] @ git+https://github.com/johnviljoen/spineax.git"
unset LD_LIBRARY_PATH
```

## CLI usage

All commands are exposed via the `topopt` CLI entrypoint.

### Train models

```bash
uv run topopt train --config ./train_configs/<config>.yaml [--save-dir ./outputs/<run_name>]
```

Required arguments:
- `--config`: Path to a training config YAML file.

Optional arguments:
- `--save-dir`: Output directory. If omitted, a timestamped directory in `./outputs` is created.

### Evaluate trained models

```bash
uv run topopt evaluate [--scale N] [--save-dir ./outputs/<run_name>] [--visualize]
```

Optional arguments:
- `--scale`: Evaluation mesh scaling factor.
- `--save-dir`: Run directory to evaluate. If omitted, the latest directory in `./outputs` is used.
- `--visualize / --no-visualize`: Enable or disable density-field visualizations.

### Run MMA baseline

```bash
uv run topopt mma [OPTIONS]
```

Common options:
- `--lx`, `--ly`: Domain size (defaults: `60.0`, `30.0`).
- `--scale`: Mesh scaling factor (default: `1.0`).
- `--bc-preset`: Boundary-condition preset name (default: `cantilever_corner`).
- `--vol-frac`: Target volume fraction (default: `0.5`).
- `--radius`: Helmholtz filter radius (default: `1.0`).
- `--max-iter`: Maximum MMA iterations (default: `200`).
- `--save-every`: Save frequency (default: `10`).
- `--print-every`: Log frequency (default: `10`).
- `--save-dir`: Output directory. If omitted, a timestamped baseline directory is created in `./outputs`.

## Acknowledgements
- [FEAX](https://github.com/Naruki-Ichihara/feax)
- [jax-fem](https://github.com/deepmodeling/jax-fem)
- [spineax](https://github.com/johnviljoen/spineax)