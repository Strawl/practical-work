# Topology Optimization with FEAX

This project performs topology optimization using **FEAX**, a finite element analysis library written in JAX.  
Follow the instructions below to set up your environment and run the training and visualization scripts.

## Installation

### Ubuntu / Debian

Install topopt:

```bash
sudo apt-get install libatlas-base-dev libblas-dev liblapack-dev libhdf5-dev libglu1-mesa libxi-dev libxmu-dev libglu1-mesa-dev libxinerama1
uv sync --group cuda12
uv run pip install --no-build-isolation --config-settings=cmake.args="-DBUILD_PBATCH_SOLVE=OFF" "spineax[cuda12] @ git+https://github.com/johnviljoen/spineax.git"
unset LD_LIBRARY_PATH
```

## Train the model

```bash
uv run ./topopt/main.py
```

## Visualize the results
```bash
uv run ./topopt/view.py --scale 3 --domain 60,30
```

### Arguments
- 	--domain: The domain used during training (e.g., 60,30).
- 	--scale: The visualization resolution (default: 1 -> equal to the domain).
- 	--dir: Directory where the outputs are stored.
If not specified, the script will automatically use the latest results from ./outputs

## Acknowledgements
- [FEAX](https://github.com/Naruki-Ichihara/feax)
- [jax-fem](https://github.com/deepmodeling/jax-fem)

## Deprecated
###  Install petsc4py (needed by jax_fem)
```bash
export MPICC=/opt/homebrew/bin/mpicc
export PETSC_DIR=$(brew --prefix petsc)
export PETSC_ARCH=real
export PEP517_BUILD_BACKEND=setuptools.build_meta
uv pip install --no-binary=petsc4py petsc4py
```