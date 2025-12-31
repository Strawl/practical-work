from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import jax


@dataclass
class MetricTracker:
    save_dir: Path = Path("./output")
    fill_invalid: bool = True

    # name -> list of per-step JAX arrays, each shape (M,)
    data: Dict[str, List[jnp.ndarray]] = field(default_factory=dict)

    # name -> expected per-step shape, e.g. (M,)
    shape: Dict[str, Tuple[int, ...]] = field(default_factory=dict)

    # name -> last logged value (for forward fill)
    last: Dict[str, jnp.ndarray] = field(default_factory=dict)

    # ------------------------ core ------------------------

    @staticmethod
    def _to_1d(values: Any) -> jnp.ndarray:
        v = jnp.asarray(values, dtype=float)
        return v.reshape((1,)) if v.ndim == 0 else v

    def log(self, name: str, values: Any) -> None:
        v = self._to_1d(values)
        if v.ndim != 1:
            raise ValueError(
                f"Metric '{name}' must be 1D (shape (M,)). Got {tuple(v.shape)}."
            )

        shp = tuple(v.shape)
        prev_shp = self.shape.get(name)
        if prev_shp is None:
            self.shape[name] = shp
        elif prev_shp != shp:
            raise ValueError(
                f"Metric '{name}' changed shape: was {prev_shp}, now {shp}."
            )

        if self.fill_invalid:
            prev = self.last.get(name)
            if prev is not None:
                v = jnp.where(jnp.isfinite(v), v, prev)

        self.data.setdefault(name, []).append(v)
        self.last[name] = v

    def steps(self, name: str) -> int:
        return len(self.data.get(name, []))

    def stack(self, name: str) -> jnp.ndarray:
        xs = self.data.get(name)
        if not xs:
            raise KeyError(f"No data logged for metric '{name}'.")
        return jnp.stack(xs, axis=0)  # (T, M)

    def all_stacked(self) -> Dict[str, jnp.ndarray]:
        return {k: self.stack(k) for k in self.data}

    def save(self, basename: str = "metrics_log") -> Path:
        npz_path = self.save_dir / Path(f"{basename}.npz")

        stacked = {k: np.asarray(self.stack(k)) for k in sorted(self.data)}
        np.savez_compressed(npz_path, **stacked)

        return npz_path

    @staticmethod
    def load(npz_path: str | Path) -> Dict[str, jnp.ndarray]:
        npz_path = Path(npz_path)
        with np.load(npz_path, allow_pickle=False) as z:
            return {k: jnp.asarray(z[k]) for k in z.files}

    # ------------------------ plotting ------------------------

    @staticmethod
    def _iter_pages(n: int, per_page: int):
        per_page = max(1, int(per_page))
        for s in range(0, n, per_page):
            yield s, min(n, s + per_page)

    @staticmethod
    def _grid_dims(n: int, ncols: int) -> Tuple[int, int]:
        ncols = max(1, int(ncols))
        nrows = max(1, math.ceil(n / ncols))
        return nrows, ncols

    def _plot_pages(
        self,
        n_items: int,
        per_page: int,
        ncols: int,
        plot_one,
        page_title,
        file_name,
        show: bool,
        save: bool,
    ) -> List[Path]:
        out_paths: List[Path] = []

        for page_idx, (s, e) in enumerate(self._iter_pages(n_items, per_page), start=1):
            n = e - s
            nrows, ncols_eff = self._grid_dims(n, ncols)

            fig, axes = plt.subplots(
                nrows=nrows,
                ncols=ncols_eff,
                figsize=(4.5 * ncols_eff, 3.2 * nrows),
            )
            axes = np.atleast_1d(axes).ravel()

            for i, item_idx in enumerate(range(s, e)):
                plot_one(axes[i], item_idx)

            for ax in axes[n:]:
                ax.axis("off")

            fig.suptitle(page_title(s, e, page_idx), y=0.95)
            fig.tight_layout()

            if save and self.save_dir is not None:
                p = self.save_dir / file_name(page_idx)
                fig.savefig(p, dpi=150, bbox_inches="tight")
                out_paths.append(p)

            if show:
                plt.show()
            else:
                plt.close(fig)

        return out_paths

    def plot_metric_across_models(
        self,
        name: str,
        model_names: Optional[List[str]] = None,
        show: bool = False,
        save: bool = True,
    ) -> List[Path]:
        hist = self.stack(name)
        T, M = map(int, hist.shape)
        if T == 0 or M == 0:
            return []

        hist_np = np.asarray(hist)
        x = np.arange(T)

        if M == 1:
            fig, ax = plt.subplots(figsize=(7.5, 4.2))
            ax.plot(x, hist_np[:, 0])
            ax.set_title(name)
            ax.set_xlabel("step")
            ax.set_ylabel(name)
            fig.tight_layout()

            out_paths: List[Path] = []
            if save and self.save_dir is not None:
                self.save_dir.mkdir(parents=True, exist_ok=True)
                p = self.save_dir / f"{name}.png"
                fig.savefig(p, dpi=150, bbox_inches="tight")
                out_paths.append(p)

            if show:
                plt.show()
            else:
                plt.close(fig)

            return out_paths

        model_names = model_names or [f"model_{m}" for m in range(M)]
        if len(model_names) != M:
            raise ValueError(f"model_names length {len(model_names)} != M={M}")

        hist_np = np.asarray(hist)
        x = np.arange(T)

        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        for m in range(M):
            ax.plot(x, hist_np[:, m], label=model_names[m])

        ax.set_title(name)
        ax.set_xlabel("step")
        ax.set_ylabel(name)

        if M <= 15:
            ax.legend(fontsize=8)

        fig.tight_layout()

        out_paths: List[Path] = []
        if save and self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            p = self.save_dir / f"{name}.png"
            fig.savefig(p, dpi=150, bbox_inches="tight")
            out_paths.append(p)

        if show:
            plt.show()
        else:
            plt.close(fig)

        return out_paths

    def plot_all_metrics_across_models(
        self,
        model_names: Optional[List[str]] = None,
        show: bool = False,
        save: bool = True,
    ) -> List[Path]:
        """
        Convenience: plot *every* logged metric, one figure per metric,
        with all models shown on each figure.
        """
        out_paths: List[Path] = []
        for name in sorted(self.data.keys()):
            out_paths += self.plot_metric_across_models(
                name=name,
                model_names=model_names,
                show=show,
                save=save,
            )
        return out_paths


class StepTimer:
    """Minimal wall-clock timer for JAX steps using perf_counter."""

    def __init__(self):
        self._t0 = None
        self.elapsed_s = None

    def start(self) -> None:
        self._t0 = time.perf_counter()
        self.elapsed_s = None

    @staticmethod
    def block_until_ready(x):
        """
        Block on a JAX array/scalar or a pytree of arrays.
        Returns x unchanged.
        """

        def _block(v):
            return v.block_until_ready() if hasattr(v, "block_until_ready") else v

        jax.tree_util.tree_map(_block, x)
        return x

    def stop(self, block_on=None) -> float:
        """
        Stop timer. If block_on is provided, we block on it before reading time.
        """
        if self._t0 is None:
            raise RuntimeError("StepTimer.stop() called before start().")

        if block_on is not None:
            self.block_until_ready(block_on)

        self.elapsed_s = time.perf_counter() - self._t0
        self._t0 = None
        return self.elapsed_s
