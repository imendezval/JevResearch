"""Versioned, task-owned global hyperparameter domains."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ....core.experiment import digest

_LR_BOUNDS = (1e-4, 1e-1)
_WD_BOUNDS = (1e-6, 1e-2)
_OPTIMIZERS = ("sgd", "adamw")
_WD_MODES = ("zero", "positive")


def _log_uniform(rng, bounds):
    return 10 ** rng.uniform(math.log10(bounds[0]), math.log10(bounds[1]))


@dataclass(frozen=True)
class CifarDomain:
    name: str

    def __post_init__(self):
        if self.name not in ("cifar-mixed-v1", "cifar-sgd-numeric-v1"):
            raise ValueError("unknown CIFAR proposal domain")

    @property
    def mixed(self):
        return self.name == "cifar-mixed-v1"

    def definition(self):
        common = {"lr": {"kind": "float", "low": _LR_BOUNDS[0], "high": _LR_BOUNDS[1], "log": True}}
        if self.mixed:
            common.update({"optimizer": {"kind": "categorical", "choices": list(_OPTIMIZERS)},
                           "wd_mode": {"kind": "categorical", "choices": list(_WD_MODES)},
                           "weight_decay": {"kind": "float", "low": _WD_BOUNDS[0],
                                            "high": _WD_BOUNDS[1], "log": True, "when": "wd_mode=positive"}})
        else:
            common["weight_decay"] = {"kind": "float", "low": _WD_BOUNDS[0],
                                      "high": _WD_BOUNDS[1], "log": True}
        return {"version": self.name, "parameters": common,
                "fixed": {"momentum": 0.9, "betas": [0.9, 0.999]}}

    @property
    def fingerprint(self):
        return digest(self.definition())

    def config(self, task, params):
        allowed = {"lr", "weight_decay"} if not self.mixed else {"optimizer", "lr", "wd_mode", "weight_decay"}
        if set(params) - allowed or "lr" not in params:
            raise ValueError("invalid domain parameter names")
        lr = params["lr"]
        if type(lr) not in (int, float) or not math.isfinite(lr) or not _LR_BOUNDS[0] <= lr <= _LR_BOUNDS[1]:
            raise ValueError("learning rate outside domain")
        if self.mixed:
            if params.get("optimizer") not in _OPTIMIZERS or params.get("wd_mode") not in _WD_MODES:
                raise ValueError("invalid categorical branch")
            if params["wd_mode"] == "zero":
                if "weight_decay" in params:
                    raise ValueError("zero branch cannot have positive weight decay")
                wd = 0.0
            else:
                wd = params.get("weight_decay")
            optimizer = params["optimizer"]
        else:
            wd, optimizer = params.get("weight_decay"), "sgd"
        if (type(wd) not in (int, float) or not math.isfinite(wd)
                or (wd != 0 and not _WD_BOUNDS[0] <= wd <= _WD_BOUNDS[1])):
            raise ValueError("weight decay outside domain")
        if not self.mixed and wd == 0:
            raise ValueError("numeric domain excludes zero weight decay")
        config = task.baseline()
        config.update({"lr": float(lr), "weight_decay": float(wd), "optimizer": optimizer})
        if optimizer == "adamw":
            config.pop("momentum")
            config.update({"beta1": 0.9, "beta2": 0.999})
        task.validate(config)
        return config

    def parameters(self, config):
        params = {"lr": config["lr"]}
        if self.mixed:
            params.update({"optimizer": config["optimizer"],
                           "wd_mode": "zero" if config["weight_decay"] == 0 else "positive"})
            if config["weight_decay"] != 0:
                params["weight_decay"] = config["weight_decay"]
        else:
            params["weight_decay"] = config["weight_decay"]
        return params

    def random_parameters(self, rng):
        params = {"lr": _log_uniform(rng, _LR_BOUNDS)}
        if self.mixed:
            params["optimizer"] = rng.choice(_OPTIMIZERS)
            params["wd_mode"] = rng.choice(_WD_MODES)
            if params["wd_mode"] == "positive":
                params["weight_decay"] = _log_uniform(rng, _WD_BOUNDS)
        else:
            params["weight_decay"] = _log_uniform(rng, _WD_BOUNDS)
        return params

    def suggest(self, trial):
        params = {"lr": trial.suggest_float("lr", *_LR_BOUNDS, log=True)}
        if self.mixed:
            params["optimizer"] = trial.suggest_categorical("optimizer", _OPTIMIZERS)
            params["wd_mode"] = trial.suggest_categorical("wd_mode", _WD_MODES)
            if params["wd_mode"] == "positive":
                params["weight_decay"] = trial.suggest_float("weight_decay", *_WD_BOUNDS, log=True)
        else:
            params["weight_decay"] = trial.suggest_float("weight_decay", *_WD_BOUNDS, log=True)
        return params

    def distributions(self, params):
        import optuna
        distributions = {"lr": optuna.distributions.FloatDistribution(*_LR_BOUNDS, log=True)}
        if self.mixed:
            distributions.update({"optimizer": optuna.distributions.CategoricalDistribution(_OPTIMIZERS),
                                  "wd_mode": optuna.distributions.CategoricalDistribution(_WD_MODES)})
        if "weight_decay" in params:
            distributions["weight_decay"] = optuna.distributions.FloatDistribution(*_WD_BOUNDS, log=True)
        return distributions
