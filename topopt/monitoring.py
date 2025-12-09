from typing import Tuple, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import jax.numpy as jnp


def forward_fill_nan_2d(arr: jnp.ndarray) -> jnp.ndarray:
    """
    Forward-fill NaNs along axis=1 (time axis) for each row (model).

    Parameters
    ----------
    arr : jnp.ndarray
        Shape (num_models, num_steps)

    Returns
    -------
    jnp.ndarray
        Same shape, with NaNs replaced by previous valid value.
        Leading NaNs are replaced by the first finite value in the row.
        Rows that are all NaN remain all NaN.
    """
    arr = jnp.asarray(arr, dtype=float)
    num_models, num_steps = arr.shape

    # We'll build updates functionally using .at
    out = arr

    for i in range(int(num_models)):
        row = out[i]
        finite_mask = jnp.isfinite(row)

        if not bool(finite_mask.any()):
            continue  # all NaN/Inf -> leave as is

        # index of first finite value
        first_idx = int(jnp.argmax(finite_mask))
        first_val = row[first_idx]

        # Fill leading invalids with first finite value
        if first_idx > 0:
            row = row.at[:first_idx].set(first_val)

        # Forward-fill remaining invalids
        for t in range(first_idx + 1, int(num_steps)):
            if not bool(jnp.isfinite(row[t])):
                row = row.at[t].set(row[t - 1])

        out = out.at[i].set(row)

    return out


def preprocess_history(
    history: Sequence[jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    history = (loss_history, lr_scale_history, lam_history, Cs_history)
    """
    loss_history, lr_scale_history, lam_history, Cs_history = history

    # Convert epochs-first -> models-first
    loss_history = loss_history.T
    lr_scale_history = lr_scale_history.T
    lam_history = lam_history.T
    Cs_history = Cs_history.T

    loss_history = forward_fill_nan_2d(loss_history)
    lr_scale_history = forward_fill_nan_2d(lr_scale_history)
    lam_history = forward_fill_nan_2d(lam_history)
    Cs_history = forward_fill_nan_2d(Cs_history)

    return loss_history, lr_scale_history, lam_history, Cs_history


def plot_single_model_history(
    i: int,
    loss_h: jnp.ndarray,
    lr_h: jnp.ndarray,
    lam_h: jnp.ndarray,
    C_h: jnp.ndarray,
    out_dir: Path,
    prefix: str = "model",
):
    """
    Save one PNG per model with 4 subplots:
    loss, lr scale, lambda, C.

    Arrays expected shape (num_models, num_epochs).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = jnp.arange(loss_h.shape[1])

    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    axs[0].plot(epochs, loss_h[i])
    axs[0].set_title(f"{prefix} {i} - Loss")
    axs[0].set_ylabel("Loss")
    axs[1].set_xlabel("Epochs")
    axs[0].grid(True)

    axs[1].plot(epochs, lr_h[i])
    axs[1].set_title(f"{prefix} {i} - LR scale")
    axs[1].set_ylabel("LR scale")
    axs[1].set_xlabel("Epochs")
    axs[1].grid(True)

    axs[2].plot(epochs, lam_h[i])
    axs[2].set_title(f"{prefix} {i} - Lambda")
    axs[2].set_ylabel("Lambda")
    axs[1].set_xlabel("Epochs")
    axs[2].grid(True)

    axs[3].plot(epochs, C_h[i])
    axs[3].set_title(f"{prefix} {i} - C")
    axs[3].set_ylabel("C")
    axs[1].set_xlabel("Epochs")
    axs[3].grid(True)

    fig.tight_layout()

    out_path = out_dir / f"{prefix}_{i}_history.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return out_path


def save_ensemble_history_plots(
    history: Sequence[jnp.ndarray],
    run_dir: Path,
    prefix: str = "model",
):
    """
    Preprocess history (forward-fill NaNs) and write one history image per model.
    Returns list of saved paths.
    """
    loss_h, lr_h, lam_h, C_h = preprocess_history(history)
    num_models = int(loss_h.shape[0])

    plots_dir = Path(run_dir) / "history_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for i in range(num_models):
        p = plot_single_model_history(
            i, loss_h, lr_h, lam_h, C_h,
            out_dir=plots_dir, prefix=prefix
        )
        paths.append(p)

    return paths