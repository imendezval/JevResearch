"""CIFAR-10 data identity, fixed split, and bounded HPO operators."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from ....core import SearchState, canonical, digest


def stratified_indices(labels, seed: int, train_per_class: int):
    groups = {}
    for index, label in enumerate(labels):
        groups.setdefault(int(label), []).append(index)
    if sorted(groups) != list(range(10)) or any(len(v) < train_per_class + 1 for v in groups.values()):
        raise ValueError("expected ten classes with enough examples for train and validation")
    rng = random.Random(seed)
    train, val = [], []
    for label in range(10):
        indices = groups[label]
        rng.shuffle(indices)
        train.extend(indices[:train_per_class])
        val.extend(indices[train_per_class:])
    return sorted(train), sorted(val)


def fixture_labels():
    return [i % 10 for i in range(100)]


def dataset_identity(root: Path, fixture: bool) -> str:
    if fixture:
        return digest({"generator": "fixture-pixels-v1", "count": 100, "labels": fixture_labels()})
    folder = root / "cifar-10-batches-py"
    files = [folder / f"data_batch_{i}" for i in range(1, 6)] + [folder / "batches.meta"]
    sha = hashlib.sha256()
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f"official CIFAR-10 training file missing: {path}")
        sha.update(path.name.encode())
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                sha.update(block)
    return sha.hexdigest()


def save_split(path: Path, labels, seed: int, train_per_class: int, dataset_sha: str):
    train, val = stratified_indices(labels, seed, train_per_class)
    manifest = {"version": "stratified-v1", "seed": seed, "dataset_sha256": dataset_sha,
                "train_indices": train, "val_indices": val}
    manifest["split_sha256"] = digest(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = json.loads(path.read_text())
        if old != manifest:
            raise ValueError("saved split differs from requested split or dataset")
    else:
        path.write_text(canonical(manifest) + "\n")
    return manifest


class CifarTask:
    objective_direction = "max"

    def __init__(self, data_dir, run_dir, *, split_seed=1729, epochs=1,
                 batch_size=256, device="cpu", download=False, fixture=False):
        if epochs < 1 or batch_size < 1:
            raise ValueError("epochs and batch size must be positive")
        if device not in ("cpu", "cuda"):
            raise ValueError("resolved device must be cpu or cuda")
        self.data_dir = Path(data_dir).resolve()
        self.run_dir = Path(run_dir).resolve()
        self.fixture = fixture
        self.name = "cifar10_fixture" if fixture else "cifar10"
        self.protocol = "fixture-v1" if fixture else "official-train-v1"
        self.eval_budget = epochs
        self.batch_size = batch_size
        self.device = device
        self.split_seed = split_seed
        if fixture:
            labels = fixture_labels()
            train_per_class = 8
        else:
            try:
                from torchvision.datasets import CIFAR10
            except ImportError as exc:
                raise RuntimeError("install the vision extra: pip install '.[vision]'") from exc
            if not download and not (self.data_dir / "cifar-10-batches-py").is_dir():
                raise FileNotFoundError("CIFAR-10 cache missing; pass --download explicitly")
            dataset = CIFAR10(str(self.data_dir), train=True, download=download)
            labels = dataset.targets
            if len(labels) != 50000:
                raise ValueError("official CIFAR-10 training set must contain 50000 images")
            train_per_class = 4500
        self.dataset_sha = dataset_identity(self.data_dir, fixture)
        self.split_path = self.run_dir / "split.json"
        self.split = save_split(self.split_path, labels, split_seed, train_per_class, self.dataset_sha)
        self.data_split = self.split["split_sha256"]

    def details(self):
        return {"dataset": self.name, "dataset_sha256": self.dataset_sha,
                "split_sha256": self.data_split, "split_seed": self.split_seed,
                "split_file": str(self.split_path), "data_dir": str(self.data_dir),
                "run_dir": str(self.run_dir), "preprocessing": "totensor-normalize-cifar-v1",
                "model": "small-convnet-v1", "metric": "final-validation-top1",
                "epochs": self.eval_budget, "batch_size": self.batch_size,
                "device": self.device, "scheduler": "none"}

    def baseline(self):
        return {"lr": 0.01, "weight_decay": 0.0001, "optimizer": "sgd",
                "momentum": 0.9, "epochs": self.eval_budget,
                "batch_size": self.batch_size, "model": "small-convnet-v1",
                "preprocessing": "totensor-normalize-cifar-v1", "scheduler": "none",
                "device": self.device}

    def validate(self, config):
        baseline = self.baseline()
        shared = set(baseline) - {"lr", "weight_decay", "optimizer", "momentum"}
        if any(config.get(key) != baseline[key] for key in shared):
            raise ValueError("fixed CIFAR training protocol changed")
        optimizer = config.get("optimizer")
        expected = shared | {"lr", "weight_decay", "optimizer"}
        expected |= {"momentum"} if optimizer == "sgd" else {"beta1", "beta2"}
        if set(config) != expected or optimizer not in ("sgd", "adamw"):
            raise ValueError("invalid optimizer-specific CIFAR config")
        lr, wd = config["lr"], config["weight_decay"]
        if not isinstance(lr, (float, int)) or not 1e-4 <= lr <= 0.1:
            raise ValueError("learning rate outside [1e-4, 0.1]")
        if not isinstance(wd, (float, int)) or not 0 <= wd <= 0.01:
            raise ValueError("weight decay outside [0, 0.01]")
        if optimizer == "sgd" and config["momentum"] != 0.9:
            raise ValueError("invalid SGD momentum")
        if optimizer == "adamw" and (config["beta1"], config["beta2"]) != (0.9, 0.999):
            raise ValueError("invalid AdamW betas")

    def proposals(self, state: SearchState):
        base = state.incumbent_config or self.baseline()
        out = []
        for factor, name in ((0.5, "lr_down"), (2, "lr_up")):
            config = dict(base)
            config["lr"] = round(base["lr"] * factor, 10)
            out.append((name, {"factor": factor}, config))
        for wd in (0.0, 0.001):
            config = dict(base)
            config["weight_decay"] = wd
            out.append(("set_weight_decay", {"value": wd}, config))
        config = dict(base)
        if base["optimizer"] == "sgd":
            config.pop("momentum")
            config.update(optimizer="adamw", beta1=0.9, beta2=0.999)
        else:
            config.pop("beta1")
            config.pop("beta2")
            config.update(optimizer="sgd", momentum=0.9)
        out.append(("swap_optimizer", {"to": config["optimizer"]}, config))
        return out
