from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


EXPERIMENT_DIR = Path(__file__).resolve().parent
SUMMARY_CSV = EXPERIMENT_DIR / "summary.csv"
NORMALIZED_DIR = EXPERIMENT_DIR / "normalized"


def trim_white(arr, threshold: float = 0.995):
    rgb = arr[..., :3] if arr.ndim == 3 else arr
    mask = (rgb < threshold).any(axis=-1) if rgb.ndim == 3 else (rgb < threshold)
    if not mask.any():
        return arr
    ys, xs = mask.nonzero()
    return arr[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


def load_image(path: Path):
    arr = mpimg.imread(path)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        alpha = arr[..., 3:4]
        arr = arr[..., :3] * alpha + (1.0 - alpha)
    return trim_white(arr)


def save_normalized_image(src: Path, dst: Path, *, width: float = 8, height: float = 4):
    arr = load_image(src)
    fig, ax = plt.subplots(figsize=(width, height), facecolor="white")
    ax.imshow(arr, interpolation="nearest", aspect="equal")
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    dst.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dst, dpi=250, facecolor="white", bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def metric_value(row: pd.Series) -> float:
    if pd.notna(row.get("compliance_feax")):
        return float(row["compliance_feax"])
    return float(row["compliance_internal"])


def parameter_text(row: pd.Series) -> str:
    settings = str(row.get("settings", ""))
    if "effective_params=" in settings:
        return f"Parameters: {settings.split('effective_params=')[1].split()[0]}"
    if row.get("method") == "SIREN":
        return "Parameters: 16897"
    if row.get("method") == "MMA":
        return "Parameters: 16200 density variables"
    if row.get("method") == "JAXTOuNN":
        return "Parameters: approx. 16.9k"
    return "Parameters: n/a"


def add_panel(ax, row: pd.Series, title: str, subtitle: str, details: list[str]):
    normalized_path = NORMALIZED_DIR / row["image_file"]
    if not normalized_path.exists():
        save_normalized_image(EXPERIMENT_DIR / row["image_file"], normalized_path)
    img = mpimg.imread(normalized_path)
    ax.imshow(img, interpolation="nearest", aspect="equal")
    ax.set_axis_off()
    compliance = metric_value(row)
    volume = float(row["volume_fraction"])
    ax.set_title(f"{title}\n{subtitle}", fontsize=30, pad=18)
    bottom_lines = [
        f"Compliance: {compliance:.3f}",
        f"Volume: {volume:.4f}",
        *details,
    ]
    ax.text(
        0.5,
        -0.14,
        "\n".join(bottom_lines),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=21,
        linespacing=1.35,
    )


def main():
    sns.set_theme(style="white", context="talk")
    df = pd.read_csv(SUMMARY_CSV)

    labels = [
        "mma_50_scale3_projected",
        "mma_50_scale15_projected",
        "jax_tounn_candidate_scale3",
        "jax_tounn_candidate_scale15",
        "siren_model_4_scale3",
        "siren_model_4_scale15",
    ]
    rows = [df.loc[df["label"] == label].iloc[0] for label in labels]
    for row in rows:
        save_normalized_image(
            EXPERIMENT_DIR / row["image_file"],
            NORMALIZED_DIR / row["image_file"],
        )

    mma3 = df.loc[df["label"] == "mma_50_scale3_projected"].iloc[0]
    mma15 = df.loc[df["label"] == "mma_50_scale15_projected"].iloc[0]
    jt3 = df.loc[df["label"] == "jax_tounn_candidate_scale3"].iloc[0]
    jt15 = df.loc[df["label"] == "jax_tounn_candidate_scale15"].iloc[0]
    s3 = df.loc[df["label"] == "siren_model_4_scale3"].iloc[0]
    s15 = df.loc[df["label"] == "siren_model_4_scale15"].iloc[0]

    fig = plt.figure(figsize=(16, 25), facecolor="white")
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.0], hspace=0.25, wspace=0.28)

    ax_mma_same = fig.add_subplot(gs[0, 0])
    add_panel(
        ax_mma_same,
        mma3,
        "MMA",
        "same-resolution evaluation",
        [
            "Evaluation mesh: 180x90",
            parameter_text(mma3),
        ],
    )
    ax_mma_refined = fig.add_subplot(gs[0, 1])
    add_panel(
        ax_mma_refined,
        mma15,
        "MMA",
        "refined evaluation",
        [
            "Evaluation mesh: 900x450",
            parameter_text(mma15),
        ],
    )

    ax_jt3 = fig.add_subplot(gs[1, 0])
    add_panel(
        ax_jt3,
        jt3,
        "JAXTOuNN",
        "same-resolution evaluation",
        [
            "Evaluation mesh: 180x90",
            parameter_text(jt3),
        ],
    )
    ax_jt15 = fig.add_subplot(gs[1, 1])
    add_panel(
        ax_jt15,
        jt15,
        "JAXTOuNN",
        "refined evaluation",
        [
            "Evaluation mesh: 900x450",
            parameter_text(jt15),
        ],
    )

    ax_s3 = fig.add_subplot(gs[2, 0])
    add_panel(
        ax_s3,
        s3,
        "SIREN",
        "same-resolution evaluation",
        [
            "Evaluation mesh: 180x90",
            parameter_text(s3),
        ],
    )
    ax_s15 = fig.add_subplot(gs[2, 1])
    add_panel(
        ax_s15,
        s15,
        "SIREN",
        "refined evaluation",
        [
            "Evaluation mesh: 900x450",
            parameter_text(s15),
        ],
    )

    png_path = EXPERIMENT_DIR / "comparison_overview.png"
    pdf_path = EXPERIMENT_DIR / "comparison_overview.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
