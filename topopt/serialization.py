import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Type

import equinox as eqx
import numpy as np
from jaxtyping import PyTree
from matplotlib.pylab import Enum
from siren import SIREN

import jax
from feax import Problem


class ModelType(str, Enum):
    SIREN = "SIREN"


MODEL_REGISTRY: Dict[ModelType, Type] = {
    ModelType.SIREN: SIREN,
}


class JSONSerializable:
    """General superclass: provides to_dict / from_dict / to_json / from_json."""

    def to_dict(self) -> Dict[str, Any]:
        # dataclasses support
        if hasattr(self, "__dataclass_fields__"):
            return asdict(self)
        raise NotImplementedError("Implement to_dict in subclasses")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JSONSerializable":
        # default works for dataclasses
        if hasattr(cls, "__dataclass_fields__"):
            return cls(**data)
        raise NotImplementedError("Implement from_dict in subclasses")

    def to_json(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "JSONSerializable":
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class TrainingParams(JSONSerializable):
    target_density: float
    penalty: float


@dataclass
class ModelInstanceConfig(JSONSerializable):
    model_type: ModelType
    model_kwargs: Dict[str, Any]
    training: TrainingParams

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
            model_kwargs=data["model_kwargs"],
            training=TrainingParams.from_dict(data["training"]),
        )


@dataclass
class ModelEnsembleConfig(JSONSerializable):
    models: List[ModelInstanceConfig] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"models": [m.to_dict() for m in self.models]}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelEnsembleConfig":
        return cls(models=[ModelInstanceConfig.from_dict(m) for m in data["models"]])


def serialize_ensemble(
    trained_models: PyTree,
    opt_states: PyTree,
    ensemble_config: ModelEnsembleConfig,
    problem: Problem,
    base_dir: str = "outputs",
    prefix: str = "model",
):
    """
    Serialize a batched collection of models, optimizer states, and their
    per-model configuration.

    Parameters
    ----------
    trained_models : PyTree
        PyTree where the leading axis indexes different models (num_models, ...).
    opt_states : PyTree
        PyTree of optimizer states with the same leading axis as trained_models.
    ensemble_config : ModelEnsembleConfig
        Contains a list of ModelInstanceConfig objects in .models.
        len(ensemble_config.models) must equal num_models.
    base_dir : str or Path, optional
        Base directory under which a timestamped run directory will be created.
    prefix : str, optional
        Prefix for the saved model filenames, e.g. 'siren' -> 'siren_0.eqx'.

    Returns
    -------
    run_dir : Path
        The directory where all files were written.
    models_list : list[PyTree]
        List of individual (unbatched) model PyTrees, one per ensemble member.
    opt_states_list : list[PyTree]
        List of individual (unbatched) optimizer state PyTrees, one per model.
    """
    base_dir = Path(base_dir)
    run_dir = base_dir / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Split trainable arrays vs. static parts for models and optimizer states
    model_arrays, model_static = eqx.partition(trained_models, eqx.is_array)
    state_arrays, state_static = eqx.partition(opt_states, eqx.is_array)

    example_leaf = jax.tree_util.tree_leaves(model_arrays)[0]
    num_models = example_leaf.shape[0]

    if hasattr(ensemble_config, "models"):
        if len(ensemble_config.models) != num_models:
            raise ValueError(
                f"num_models ({num_models}) != len(ensemble_config.models) "
                f"({len(ensemble_config.models)})"
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

        model_path = run_dir / f"{prefix}_{i}.eqx"
        state_path = run_dir / f"opt_state_{i}.eqx"
        cfg_path = run_dir / f"{prefix}_{i}_config.json"

        eqx.tree_serialise_leaves(model_path, model_i)
        eqx.tree_serialise_leaves(state_path, state_i)

        if hasattr(ensemble_config, "models"):
            instance_cfg = ensemble_config.models[i]

            cfg_dict = instance_cfg.to_dict()
            cfg_dict["weights_file"] = str(model_path.name)
            cfg_dict["opt_state_file"] = str(state_path.name)

            with open(cfg_path, "w") as f:
                json.dump(cfg_dict, f, indent=2)

    return run_dir, models_list, opt_states_list


def create_models(
    ensemble_cfg: ModelEnsembleConfig,
    rng_key: jax.random.PRNGKey,
):
    configs = ensemble_cfg.models
    num_models = len(configs)

    if num_models == 0:
        raise ValueError("No models in ensemble config")

    model_types = [cfg.model_type for cfg in configs]
    first_type = model_types[0]
    if not all(mt == first_type for mt in model_types):
        raise ValueError(
            "All models in one ensemble must share the same model_type "
            "for batched construction. Got: "
            + ", ".join(mt.value for mt in model_types)
        )

    model_cls = MODEL_REGISTRY[first_type]

    keys = jax.random.split(rng_key, num_models)

    models = [model_cls(rng_key=k, **cfg.model_kwargs) for k, cfg in zip(keys, configs)]

    model_batch = jax.tree_util.tree_map(lambda *xs: np.stack(xs), *models)

    target_densities = np.array([cfg.training.target_density for cfg in configs])
    penalties = np.array([cfg.training.penalty for cfg in configs])

    return model_batch, target_densities, penalties


def load_model_from_config(cfg_path: Path, base_dir: Path):
    """Load a trained model using its per-model config JSON.

    Assumes the config was created by `serialize_ensemble`, i.e. it contains:
      - model_type
      - model_kwargs
      - training
      - weights_file
    """
    with cfg_path.open("r") as f:
        cfg = json.load(f)

    model_type_str = cfg.get("model_type")
    try:
        model_type = ModelType(model_type_str)
    except ValueError:
        raise NotImplementedError(
            f"model_type '{model_type_str}' is not supported yet in view_models.py"
        )

    model_kwargs = cfg.get("model_kwargs", {})
    training_cfg = cfg.get("training", {})

    weights_file = cfg.get("weights_file", None)
    if weights_file is None:
        base = cfg_path.stem.replace("_config", "")
        weights_file = f"{base}.eqx"

    weights_path = base_dir / weights_file

    try:
        model_cls = MODEL_REGISTRY[model_type]
    except KeyError:
        raise NotImplementedError(
            f"model_type '{model_type.value}' is not supported yet in view_models.py"
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
