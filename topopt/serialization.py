from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type, TypeVar, Union

import equinox as eqx
import jax.numpy as jnp
import yaml
from jaxtyping import PyTree

import jax
from topopt.siren import SIREN


class ModelType(str, Enum):
    SIREN = "SIREN"


MODEL_REGISTRY: Dict[ModelType, Type] = {
    ModelType.SIREN: SIREN,
}

T = TypeVar("T", bound="ConfigSerializable")


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _as_path(p: Union[str, Path]) -> Path:
    return p if isinstance(p, Path) else Path(p)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    _safe_mkdir(path.parent)
    path.write_text(text, encoding="utf-8")


class ConfigSerializable:
    """
    Dataclass-friendly config IO.

    Notes:
    - The default to_dict uses dataclasses.asdict().
    - For nested custom types (enums, nested dataclasses, list of configs),
      override to_dict/from_dict in that dataclass.
    - Supports YAML
    """

    def to_dict(self) -> Dict[str, Any]:
        if hasattr(self, "__dataclass_fields__"):
            return asdict(self)
        raise NotImplementedError("Implement to_dict in subclasses")

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        if hasattr(cls, "__dataclass_fields__"):
            return cls(**data)  # type: ignore[arg-type]
        raise NotImplementedError("Implement from_dict in subclasses")

    def to_yaml(self, path: Union[str, Path]) -> Path:
        path = _as_path(path)
        payload = self.to_dict()
        _write_text(path, yaml.safe_dump(payload, sort_keys=False))
        return path

    @classmethod
    def from_yaml(cls: Type[T], path: Union[str, Path]) -> T:
        path = _as_path(path)
        data = yaml.safe_load(_read_text(path))
        if not isinstance(data, dict):
            raise ValueError(
                f"YAML config must be a mapping at top-level. Got: {type(data)}"
            )
        return cls.from_dict(data)


@dataclass
class ModelTrainingParams(ConfigSerializable):
    """Per-model (per-ensemble-member) training settings."""

    target_density: float
    penalty: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelTrainingParams":
        return cls(
            target_density=float(data["target_density"]),
            penalty=float(data["penalty"]),
        )


@dataclass
class ModelInstanceConfig(ConfigSerializable):
    model_type: ModelType
    model_kwargs: Dict[str, Any]
    training: ModelTrainingParams

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_type.value,
            "model_kwargs": self.model_kwargs,
            "training": self.training.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelInstanceConfig":
        return cls(
            model_type=ModelType(data["model_type"]),
            model_kwargs=dict(data.get("model_kwargs", {})),
            training=ModelTrainingParams.from_dict(data["training"]),
        )


def _require(d: Dict[str, Any], key: str, ctx: str = "") -> Any:
    if key not in d:
        prefix = f"{ctx}: " if ctx else ""
        raise KeyError(f"{prefix}Missing required config key '{key}'")
    return d[key]


@dataclass
class PlateauConfig(ConfigSerializable):
    patience: int
    cooldown: int
    factor: float
    rtol: float
    accumulation_size: int

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PlateauConfig":
        ctx = cls.__name__
        return cls(
            patience=int(_require(d, "patience", ctx)),
            cooldown=int(_require(d, "cooldown", ctx)),
            factor=float(_require(d, "factor", ctx)),
            rtol=float(_require(d, "rtol", ctx)),
            accumulation_size=int(_require(d, "accumulation_size", ctx)),
        )


@dataclass
class TrainingHyperparams(ConfigSerializable):
    """Run-level training hyperparameters (shared across all models)."""

    num_iterations: int
    lr: float
    jitted_coords: bool
    Lx: float
    Ly: float
    scale: int
    problem_type: str
    helmholtz_radius: float

    grad_clip_norm: float
    max_consecutive_errors: int

    train_rng_seed: int
    model_rng_seed: int

    plateau: PlateauConfig

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["plateau"] = self.plateau.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingHyperparams":
        ctx = cls.__name__
        d = dict(d)

        plateau_raw = _require(d, "plateau", ctx)
        if not isinstance(plateau_raw, dict):
            raise TypeError(
                f"{ctx}: 'plateau' must be a mapping/dict, got {type(plateau_raw)}"
            )

        return cls(
            num_iterations=int(_require(d, "num_iterations", ctx)),
            lr=float(_require(d, "lr", ctx)),
            jitted_coords=bool(_require(d, "jitted_coords", ctx)),
            Lx=float(_require(d, "Lx", ctx)),
            Ly=float(_require(d, "Ly", ctx)),
            scale=int(_require(d, "scale", ctx)),
            problem_type=str(_require(d, "problem_type", ctx)),
            helmholtz_radius=float(_require(d, "helmholtz_radius", ctx)),
            grad_clip_norm=float(_require(d, "grad_clip_norm", ctx)),
            max_consecutive_errors=int(_require(d, "max_consecutive_errors", ctx)),
            train_rng_seed=int(_require(d, "train_rng_seed", ctx)),
            model_rng_seed=int(_require(d, "model_rng_seed", ctx)),
            plateau=PlateauConfig.from_dict(plateau_raw),
        )


@dataclass
class TrainingConfig(ConfigSerializable):
    """
    Top-level config: models + run-level training hyperparams.
    Explicit: no defaults.
    """

    models: List[ModelInstanceConfig]
    training: TrainingHyperparams

    def to_dict(self) -> Dict[str, Any]:
        return {
            "models": [m.to_dict() for m in self.models],
            "training": self.training.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        ctx = cls.__name__

        models_raw = _require(data, "models", ctx)
        if not isinstance(models_raw, list):
            raise TypeError(f"{ctx}: 'models' must be a list, got {type(models_raw)}")

        training_raw = _require(data, "training", ctx)
        if not isinstance(training_raw, dict):
            raise TypeError(
                f"{ctx}: 'training' must be a mapping/dict, got {type(training_raw)}"
            )

        return cls(
            models=[ModelInstanceConfig.from_dict(m) for m in models_raw],
            training=TrainingHyperparams.from_dict(training_raw),
        )


def serialize_ensemble(
    trained_models: PyTree,
    opt_states: PyTree,
    train_cfg: TrainingConfig,
    save_dir: Path,
    prefix: str = "model",
):
    """
    Serialize a batched collection of models, optimizer states, and their configs.

    Writes:
      - {prefix}_{i}.eqx
      - opt_state_{i}.eqx
      - {prefix}_{i}_config.(yaml)  (contains model_type, model_kwargs, training, plus filenames)
      - training_config_snapshot.(yaml)  (full TrainingConfig snapshot)

    Returns:
      models_list, opt_states_list
    """
    snapshot_path = save_dir / "training_config_snapshot.yaml"
    train_cfg.to_yaml(snapshot_path)

    model_arrays, model_static = eqx.partition(trained_models, eqx.is_array)
    state_arrays, state_static = eqx.partition(opt_states, eqx.is_array)

    example_leaf = jax.tree_util.tree_leaves(model_arrays)[0]
    num_models = int(example_leaf.shape[0])

    if len(train_cfg.models) != num_models:
        raise ValueError(
            f"num_models ({num_models}) != len(train_cfg.models) ({len(train_cfg.models)})"
        )

    models_list = []
    opt_states_list = []

    for i in range(num_models):
        model_arrays_i = jax.tree_util.tree_map(lambda x: x[i], model_arrays)
        state_arrays_i = jax.tree_util.tree_map(lambda x: x[i], state_arrays)

        model_i = eqx.combine(model_arrays_i, model_static)
        state_i = eqx.combine(state_arrays_i, state_static)

        models_list.append(model_i)
        opt_states_list.append(state_i)

        model_path = save_dir / f"{prefix}_{i}.eqx"
        state_path = save_dir / f"opt_state_{i}.eqx"
        cfg_path = save_dir / f"{prefix}_{i}_config.yaml"

        eqx.tree_serialise_leaves(model_path, model_i)
        eqx.tree_serialise_leaves(state_path, state_i)

        instance_cfg = train_cfg.models[i]
        cfg_dict = instance_cfg.to_dict()
        cfg_dict["weights_file"] = str(model_path.name)
        cfg_dict["opt_state_file"] = str(state_path.name)

        _write_text(cfg_path, yaml.safe_dump(cfg_dict, sort_keys=False))

    return models_list, opt_states_list


def create_models(
    train_cfg: TrainingConfig,
    rng_key: jax.random.PRNGKey,
) -> Tuple[PyTree, jnp.ndarray, jnp.ndarray]:
    """
    Returns:
      model_batch: PyTree with leading axis (num_models, ...)
      target_densities: (num_models,)
      penalties: (num_models,)
    """
    configs = train_cfg.models
    num_models = len(configs)
    if num_models == 0:
        raise ValueError("No models in training config")

    model_types = [cfg.model_type for cfg in configs]
    first_type = model_types[0]
    if not all(mt == first_type for mt in model_types):
        raise ValueError(
            "All models in one ensemble must share the same model_type for batched construction. Got: "
            + ", ".join(mt.value for mt in model_types)
        )

    model_cls = MODEL_REGISTRY[first_type]
    keys = jax.random.split(rng_key, num_models)

    models = [model_cls(rng_key=k, **cfg.model_kwargs) for k, cfg in zip(keys, configs)]

    # Keep semantics: stack leaf-wise across models to create a batched PyTree.
    model_batch = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *models)

    target_densities = jnp.asarray(
        [cfg.training.target_density for cfg in configs], dtype=float
    )
    penalties = jnp.asarray([cfg.training.penalty for cfg in configs], dtype=float)

    return model_batch, target_densities, penalties


def _load_cfg_dict(cfg_path: Path) -> Dict[str, Any]:
    ext = cfg_path.suffix.lower()
    if ext in (".yaml", ".yml"):
        data = yaml.safe_load(_read_text(cfg_path))
        if not isinstance(data, dict):
            raise ValueError(
                f"YAML per-model config must be a mapping. Got: {type(data)}"
            )
        return data
    raise ValueError(f"Unsupported per-model config extension '{cfg_path.suffix}'.")


def load_model_from_config(cfg_path: Path, base_dir: Path):
    """
    Load a trained model using its per-model config (YAML).

    Assumes the config was created by `serialize_ensemble`, i.e. it contains:
      - model_type
      - model_kwargs
      - training
      - weights_file (optional; inferred if missing)
    """
    cfg_path = _as_path(cfg_path)
    base_dir = _as_path(base_dir)

    cfg = _load_cfg_dict(cfg_path)

    model_type_str = cfg.get("model_type")
    try:
        model_type = ModelType(model_type_str)
    except Exception:
        raise NotImplementedError(
            f"model_type '{model_type_str}' is not supported yet in load_model_from_config"
        )

    model_kwargs = cfg.get("model_kwargs", {}) or {}
    training_cfg = cfg.get("training", {}) or {}

    weights_file = cfg.get("weights_file", None)
    if weights_file is None:
        base = cfg_path.stem.replace("_config", "")
        weights_file = f"{base}.eqx"

    weights_path = base_dir / weights_file

    try:
        model_cls = MODEL_REGISTRY[model_type]
    except KeyError:
        raise NotImplementedError(
            f"model_type '{model_type.value}' is not supported yet in load_model_from_config"
        )

    rng = jax.random.PRNGKey(0)
    model_dummy = model_cls(
        rng_key=rng,
        **model_kwargs,
    )
    model = eqx.tree_deserialise_leaves(weights_path, model_dummy)

    if isinstance(training_cfg, dict):
        target_density = training_cfg.get("target_density")
        penalty = training_cfg.get("penalty")
    else:
        target_density = getattr(training_cfg, "target_density", None)
        penalty = getattr(training_cfg, "penalty", None)

    return model, target_density, penalty, cfg
