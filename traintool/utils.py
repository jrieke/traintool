from typing import Union
from pathlib import Path
from datetime import datetime


def filter_dict(d, keys):
    """Returns a dictionary which contains only the keys in the list `keys`."""
    return {k: v for k, v in d.items() if k in keys}


class DummyWith:
    """Placeholder for object in a with statement."""

    def __enter__(self):
        pass

    def __exit__(self, model_type, value, tb):
        pass


class DummyExperiment:
    """Placeholder for comet_ml.Experiment object if comet is not connected."""

    def __init__(self):
        pass

    def train(self):
        return DummyWith()

    def test(self):
        return DummyWith()

    def log_metric(self, name, value, step=None):
        pass

    def end(self):
        pass


def timestamp() -> str:
    """Return a string timestamp."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
