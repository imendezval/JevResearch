"""Switch between the benchmark's two valid optimizer parameter sets."""


class SwapOptimizer:
    name = "swap_optimizer"

    def parameters(self):
        return {}

    def apply(self, config):
        result = dict(config)
        if result["optimizer"] == "sgd":
            result.pop("momentum")
            result.update(optimizer="adamw", beta1=0.9, beta2=0.999)
        elif result["optimizer"] == "adamw":
            result.pop("beta1")
            result.pop("beta2")
            result.update(optimizer="sgd", momentum=0.9)
        else:
            raise ValueError("unsupported optimizer")
        return result
