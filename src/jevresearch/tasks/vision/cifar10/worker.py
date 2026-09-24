"""Isolated, deterministic training process for the CIFAR task."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from ....core import ExperimentResult, ExperimentSpec, canonical, digest
from .model import make_model
from .task import dataset_identity, fixture_labels


def _datasets(details, split, seed):
    import numpy as np
    import torch
    from torch.utils.data import Dataset, Subset
    from torchvision import transforms
    from torchvision.datasets import CIFAR10

    normalizer = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                          (0.247, 0.243, 0.261))])
    if details["dataset"] == "cifar10_fixture":
        class Fixture(Dataset):
            def __init__(self):
                rng = np.random.default_rng(8243)
                self.images = rng.integers(0, 128, (100, 32, 32, 3), dtype=np.uint8)
                self.labels = fixture_labels()
                for i, label in enumerate(self.labels):
                    self.images[i, :, :, label % 3] += 50

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, index):
                from PIL import Image
                return normalizer(Image.fromarray(self.images[index])), self.labels[index]
        dataset = Fixture()
    else:
        dataset = CIFAR10(details["data_dir"], train=True, download=False,
                          transform=normalizer)
        if len(dataset) != 50000:
            raise ValueError("unexpected CIFAR training count")
    return Subset(dataset, split["train_indices"]), Subset(dataset, split["val_indices"])


def train(payload, checkpoint_path=None):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    spec = ExperimentSpec(**payload["spec"])
    details, split = payload["details"], payload["split"]
    saved = json.loads(Path(details["split_file"]).read_text())
    if saved != split or digest({k: v for k, v in split.items() if k != "split_sha256"}) != spec.data_split:
        raise ValueError("split manifest does not match trial spec")
    if dataset_identity(Path(details["data_dir"]), details["dataset"] == "cifar10_fixture") != split["dataset_sha256"]:
        raise ValueError("dataset content changed since session creation")
    if spec.task != details["dataset"] or spec.eval_budget != details["epochs"]:
        raise ValueError("task or training budget changed")
    config = spec.config
    if any(config[key] != details[key] for key in ("epochs", "batch_size", "model", "preprocessing", "scheduler", "device")):
        raise ValueError("trial config differs from frozen protocol")
    random.seed(spec.seed)
    np.random.seed(spec.seed)
    torch.manual_seed(spec.seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    train_set, val_set = _datasets(details, split, spec.seed)
    generator = torch.Generator().manual_seed(spec.seed)
    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True,
                              generator=generator, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=config["batch_size"], shuffle=False,
                            num_workers=0)
    model = make_model().to(device)
    if config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"],
                                    weight_decay=config["weight_decay"],
                                    momentum=config["momentum"])
    elif config["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                      weight_decay=config["weight_decay"],
                                      betas=(config["beta1"], config["beta2"]))
    else:
        raise ValueError("unknown optimizer")
    training_start = time.monotonic()
    train_loss = 0.0
    train_count = 0
    for _ in range(config["epochs"]):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * labels.numel()
            train_count += labels.numel()
    training_seconds = time.monotonic() - training_start
    model.eval()
    val_loss = 0.0
    correct = total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            val_loss += F.cross_entropy(logits, labels, reduction="sum").item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.numel()
    metrics = {"train_loss": train_loss / train_count,
               "validation_loss": val_loss / total,
               "validation_accuracy": correct / total,
               "train_count": train_count // config["epochs"], "validation_count": total,
               "training_seconds": training_seconds,
               "hardware": {"device": str(device), "cpu": platform.processor() or platform.machine(),
                            "torch": torch.__version__, "cuda_name":
                            torch.cuda.get_device_name(device) if device.type == "cuda" else None},
               "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device)
               if device.type == "cuda" else None,
               "gpu_active_seconds": None}
    artifacts = ()
    if checkpoint_path is not None:
        temporary = checkpoint_path.with_suffix(".tmp")
        torch.save(model.state_dict(), temporary)
        temporary.replace(checkpoint_path)
        checkpoint_manifest = {"sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                               "spec_fingerprint": spec.fingerprint,
                               "protocol": spec.protocol, "split": spec.data_split,
                               "model_version": config["model"],
                               "torch": torch.__version__, "numpy": np.__version__}
        path = checkpoint_path.with_name("checkpoint-manifest.json")
        temp_manifest = path.with_suffix(".tmp")
        temp_manifest.write_text(canonical(checkpoint_manifest) + "\n")
        temp_manifest.replace(path)
        artifacts = (f"trial-{payload['trial_id']}/model.pt",
                     f"trial-{payload['trial_id']}/checkpoint-manifest.json")
    return ExperimentResult("completed", metrics["validation_accuracy"], metrics,
                            time.monotonic() - training_start, artifacts=artifacts)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    start = time.monotonic()
    payload = json.loads(Path(args.input).read_text())
    try:
        checkpoint = Path(args.output).with_name("model.pt") if payload.get("checkpoint_policy") == "best" else None
        result = train(payload, checkpoint)
    except Exception as exc:
        traceback.print_exc()
        result = ExperimentResult("failed", None, {}, time.monotonic() - start,
                                  type(exc).__name__, str(exc))
    output = Path(args.output)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(canonical(asdict(result)) + "\n")
    temporary.replace(output)
    if "trial_id" in payload:
        manifest = {"trial_id": payload["trial_id"],
                    "spec_fingerprint": payload["spec_fingerprint"],
                    "input_sha256": hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
                    "result_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                    "worker_wall_seconds": time.monotonic() - start}
        path = output.with_name("result-manifest.json")
        temporary = path.with_suffix(".tmp")
        temporary.write_text(canonical(manifest) + "\n")
        temporary.replace(path)


if __name__ == "__main__":
    main()
