from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


EXPERIMENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = EXPERIMENT_DIR.parent.parent
BASELINE_DIR = ROOT_DIR / "outputs/06-23_17-56-03_baseline_50_iterations"
SIREN_DIR = ROOT_DIR / "outputs/06-23_19-04-30_neural_field_train_simple-Tounn_best"
JAXTOUNN_DIR = (ROOT_DIR / "../JAXTOuNN").resolve()
TRAJECTORY_CSV = EXPERIMENT_DIR / "compliance_trajectories.csv"
PNG_PATH = EXPERIMENT_DIR / "compliance_trajectories.png"
PDF_PATH = EXPERIMENT_DIR / "compliance_trajectories.pdf"


def load_baseline_history() -> pd.DataFrame:
    df = pd.read_csv(BASELINE_DIR / "history.csv")
    return pd.DataFrame(
        {
            "source": "MMA",
            "iteration": df["iteration"].astype(int),
            "compliance": df["objective"].astype(float),
            "volume_fraction": df["volume"].astype(float),
        }
    )


def load_siren_history(model_index: int = 3) -> pd.DataFrame:
    metrics = np.load(SIREN_DIR / "metrics_log.npz")
    compliance = np.asarray(metrics["true_compliance"][:, model_index], dtype=float)
    volume_fraction = np.asarray(metrics["volume_fraction"][:, model_index], dtype=float)
    return pd.DataFrame(
        {
            "source": "SIREN",
            "iteration": np.arange(1, compliance.shape[0] + 1, dtype=int),
            "compliance": compliance,
            "volume_fraction": volume_fraction,
        }
    )


def load_jax_tounn_history() -> pd.DataFrame:
    sys.path.insert(0, str(JAXTOUNN_DIR))
    try:
        from export_density_vtu import build_problem
    finally:
        sys.path.pop(0)

    warnings.filterwarnings("ignore", message="Matrix is nearly singular.*")
    warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive.*")

    tounn, opt_params = build_problem(str(JAXTOUNN_DIR / "config.txt"))
    # Prevent GUI updates and avoid snapshot writes outside the workspace.
    tounn.FE.mesh.plotFieldOnMesh = lambda *args, **kwargs: None
    history = tounn.optimizeDesign(opt_params, disableDisplay=False)

    return pd.DataFrame(
        {
            "source": "JAXTOuNN",
            "iteration": np.asarray(history["epoch"], dtype=int) + 1,
            "compliance": np.asarray(history["J"], dtype=float),
            "volume_fraction": np.asarray(history["vf"], dtype=float),
        }
    )


def make_plot(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    palette = {
        "MMA": "#1b9e77",
        "JAXTOuNN": "#d95f02",
        "SIREN": "#7570b3",
    }

    fig, ax = plt.subplots(1, 1, figsize=(10, 6), facecolor="white")

    sns.lineplot(
        data=df,
        x="iteration",
        y="compliance",
        hue="source",
        style="source",
        markers=True,
        dashes=False,
        linewidth=2.8,
        markersize=8,
        palette=palette,
        ax=ax,
    )
    ax.set_yscale("log")
    ax.set_title("Compliance")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Compliance")
    ax.set_xlim(1, 50)
    ax.legend(title="")

    fig.suptitle("Compliance Reduction Over 50 Iterations", fontsize=24, y=1.02)
    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(PDF_PATH, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    baseline = load_baseline_history()
    siren = load_siren_history(model_index=3)
    jax_tounn = load_jax_tounn_history()

    df = pd.concat([baseline, jax_tounn, siren], ignore_index=True)
    df.to_csv(TRAJECTORY_CSV, index=False)
    make_plot(df)
    print(TRAJECTORY_CSV)
    print(PNG_PATH)
    print(PDF_PATH)


if __name__ == "__main__":
    main()
