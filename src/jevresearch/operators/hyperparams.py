"""Fixed hyperparameter moves used by the CIFAR benchmark."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ScaleLearningRate:
    factor: float

    @property
    def name(self):
        return "lr_up" if self.factor > 1 else "lr_down"

    def parameters(self):
        return {"factor": self.factor}

    def apply(self, config):
        result = dict(config)
        result["lr"] = round(config["lr"] * self.factor, 10)
        return result


@dataclass(frozen=True)
class SetWeightDecay:
    value: float
    name = "set_weight_decay"

    def parameters(self):
        return {"value": self.value}

    def apply(self, config):
        result = dict(config)
        result["weight_decay"] = self.value
        return result
