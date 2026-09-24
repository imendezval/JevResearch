"""Compatibility entry point for existing worker commands."""

from .tasks.vision.cifar10.worker import _datasets, main, train

if __name__ == "__main__":
    main()
