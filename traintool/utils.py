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

    def validate(self):
        return DummyWith()

    def log_metric(self, name, value, step=None):
        pass

    def end(self):
        pass


def timestamp() -> str:
    """Return a string timestamp."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def format_timedelta(delta) -> str:
    """Formats timedelta to x days, x h, x min, x s."""
    s = delta.total_seconds()
    days, remainder = divmod(s, 86400)
    hours, remainder = divmod(s, 3600)
    mins, secs = divmod(remainder, 60)
    
    days = int(days)
    hours = int(hours)
    mins = int(mins)
    secs = int(secs)

    output = f"{secs} s"
    if mins:
        output = f"{mins} min, " + output
    if hours:
        output = f"{hours} h, " + output
    if days:
        output = f"{days} days, " + output
    return output
