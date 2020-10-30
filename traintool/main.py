from comet_ml import Experiment
from typing import Union, Type
from pathlib import Path
import yaml
from tensorboardX import SummaryWriter
import tempfile

from . import utils
from . import image_classification
from .model_wrapper import ModelWrapper


comet_config = {}
# TODO: Add a method to change the project dir.
project_dir = Path.cwd() / "traintool-experiments"


def connect_comet(api_key: str = None, project_name: str = None) -> None:
    """Connect comet.ml account to traintool (required to track metrics)."""
    comet_config["api_key"] = api_key
    comet_config["project_name"] = project_name


def _resolve_model(model_name: str) -> Type[ModelWrapper]:
    """Return class of model wrapper that is used for model_name."""
    # TODO: Maybe change the structure so that each model has its unique class, by
    #   subclassing a common class.
    if model_name in [
        "simple-cnn",
        "resnet18",
        "alexnet",
        "vgg16",
        "squeezenet",
        "densenet",
        "inception",
        "googlenet",
        "shufflenet",
        "mobilenet",
        "resnext50_32x4d",
        "wide_resnet50_2",
        "mnasnet",
    ]:
        return image_classification.TorchImageClassificationWrapper
    elif model_name in [
        "random-forest",
        "gradient-boosting",
        "gaussian-process",
        "logistic-regression",
        "sgd",
        "perceptron",
        "passive-aggressive",
        "gaussian-nb",
        "k-neighbors",
        "mlp",
        "svc",
        "linear-svc",
        "decision-tree",
        "extra-tree",
    ]:
        return image_classification.SklearnImageClassificationWrapper
    else:
        raise ValueError(f"Model not recognized: {model_name}")


def _update_config(default_config: dict, config: dict) -> dict:
    """Update values in default_config with values in config"""
    final_config = default_config.copy()
    for key, value in config.items():
        if key not in final_config:
            raise ValueError(
                "config contains a parameter that is not supported for this model: "
                f"{key}"
            )
        else:
            final_config[key] = value
    return final_config


def _write_info_file(out_dir: Path, model_name: str, config: dict) -> None:
    """Create a file info.yml in out_dir that contains some information about the run"""
    # TODO: Add more stuff, e.g. start time, status, machine configuration.
    info = {"model_name": model_name, "config": config}
    with (out_dir / "info.yml").open("w") as f:
        yaml.dump(info, f)


def _read_info_file(out_dir: Path) -> None:
    """Read the file out_dir / info.yml and return its content."""
    # TODO: Add more stuff, e.g. start time, status, machine configuration.
    with (out_dir / "info.yml").open("r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def _create_comet_experiment(
    config: dict = None, dry_run: bool = False
) -> Union[Experiment, utils.DummyExperiment]:
    """
    Creates a comet_ml.Experiment object, or a dummy replacement object. 
    
    If traintool.connect_comet was not called, this returns a dummy object instead, 
    which offers the same methods as comet_ml.Experiment, but does nothing. This 
    function should be called right before the experiment starts because comet_ml 
    prints some logs.
    """
    if "api_key" in comet_config:
        experiment = Experiment(
            api_key=comet_config["api_key"],
            project_name=comet_config["project_name"],
            disabled=dry_run,  # does not write to server if dry_run is True
            # TODO: Even create experiment on dry run?
        )
        experiment.log_parameters(config)
    else:
        experiment = utils.DummyExperiment()
    return experiment


def _create_tensorboard_writer(out_dir: Path, write_to_disk: bool = True):
    """Returns a writer for tensorboard that logs to out_dir"""
    return SummaryWriter(logdir=(out_dir / "tb").resolve(), write_to_disk=write_to_disk)


def train(
    model_name: str,
    train_data,
    val_data=None,
    test_data=None,
    config: dict = None,
    save: bool = True,
    out_dir: Union[Path, str] = None,
    dry_run: bool = False,
) -> ModelWrapper:
    """
    Starts a training run and returns a wrapper around the model.

    Args:
        model_name (str): Name of the model to train
        train_data: Training data. Multiple formats available, see docs.
        val_data (optional): Validation data. Multiple formats available, see docs. 
            Defaults to None.
        test_data (optional): Test data. Multiple formats available, see docs. 
            Defaults to None.
        config (dict, optional): Configuration of hyperparameters. If None (default), 
            some default hyperparameters for each model are used.
        save (bool, optional): Whether the model, logs and tensorboard metrics will be 
            saved to disk and (if connected) to comet_ml. Defaults to True.
        out_dir (Union[Path, str], optional): The directory where to store the model and 
            logs. If None (default), a timestamped directory is automatically created in 
            ./traintool-experiments.
        dry_run (bool, optional): If True, the model will only be trained for one batch 
            to check that everything works. This will still save the model to 
            disk; use save=False to prevent this. Defaults to False.
        
    Returns:
        ModelWrapper: A wrapper around the original model
    """

    # TODO: Disabled for now, so that we can pass parameters more dynamically (e.g. pass
    #   along sklearn model params without specifying them explicitly), think about if
    #   this is still required.
    # Get default config and fill up with values from config (checking that all keys
    # are correct)
    # default_config = model_wrapper_class.default_config(model_name)
    # if config is None:
    #     config = default_config
    # else:
    #     config = _update_config(default_config, config)

    with tempfile.TemporaryDirectory() as tmp_dir:  # only used when save=True

        if config is None:
            config = {}

        # Create out_dir and file with some general information.
        experiment_name = f"{utils.timestamp()}_{model_name}"
        if save:
            if out_dir is None:
                out_dir = project_dir / experiment_name
                out_dir.mkdir(parents=True, exist_ok=False)
            else:
                out_dir = Path(out_dir)
        else:
            # TODO: For convenience, we are just making out_dir a temporary directory
            #   here, which will be discarded. Instead, we shouldn't save anything at
            #   all because it might be more performant.
            out_dir = Path(tmp_dir)
        _write_info_file(out_dir, model_name=model_name, config=config)

        # Create model wrapper based on model_name (checking that model_name is valid)
        model_wrapper_class = _resolve_model(model_name)
        model_wrapper = model_wrapper_class(model_name, config, out_dir)

        # Print some info
        # print("=" * 29, "traintool experiment", "=" * 29)
        print("  traintool experiment  ".center(80, "="))
        print("Name:".ljust(15), experiment_name)
        print("Model name:".ljust(15), model_name)
        print("Configuration:".ljust(15), config)
        print("Output dir:".ljust(15), out_dir)
        if "api_key" in comet_config:
            print(
                "Logging to comet.ml",
                f"(project: {comet_config['project_name']})"
                if comet_config["project_name"] is not None
                else "",
            )
        print("=" * 80)
        if dry_run:
            print(">>> THIS IS JUST A DRY RUN <<<")
            print()

        # Create tensorboard writer
        writer = _create_tensorboard_writer(out_dir=out_dir)

        # Create comet.ml experiment (or dummy object if comet is not used).
        # This has to be done right before training because it prints some stuff.
        experiment = _create_comet_experiment(config=config, dry_run=save)

        # Start training the model
        model_wrapper._train(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            writer=writer,
            experiment=experiment,
            dry_run=dry_run,
        )

        # End experiment
        experiment.end()
        writer.close()
        print()
        print("=" * 80)
        print()
        print("Finished!")

        # Add end time and run time to out_dir / info.yml
        # TODO

        return model_wrapper


def load(name_or_dir: Union[str, Path]) -> ModelWrapper:
    """
    Load a model that was trained previously.

    Args:
        name_or_dir (Union[str, Path]): The name of the experiment (will search for 
            model files in ./{name} and ./traintool-experiments/{name}) or the complete 
            path to the output directory.

    Returns:
        ModelWrapper: The loaded model
    """

    # Find output dir, checking that it exists
    if Path(name_or_dir).is_dir():  # path = name_or_dir
        out_dir = Path(name_or_dir)
    elif (Path.cwd() / name_or_dir).is_dir():  # path = ./name_or_dir
        out_dir = Path.cwd() / name_or_dir
    elif (
        Path.cwd() / "traintool-experiments" / name_or_dir
    ).is_dir():  # path = ./traintool-experiments/name_or_dir
        out_dir = Path.cwd() / "traintool-experiments" / name_or_dir
    else:
        raise FileNotFoundError(f"Could not find experiment or path: {name_or_dir}")

    # Read model_name from out_dir / info.yml
    info = _read_info_file(out_dir)
    model_name = info["model_name"]

    # Resolve model wrapper class and call load method
    model_wrapper_class = _resolve_model(model_name)
    model_wrapper = model_wrapper_class(model_name, info["config"], out_dir)
    model_wrapper._load()
    return model_wrapper


# TODO: Maybe change this to more general method `model_info`.
# def default_config(model_name: str) -> dict:
#     # TODO: Get actual values here.
#     """
#     Returns the default hyperparameter configuration for a model.

#     Args:
#         model_name (str): The model.

#     Returns:
#         dict: Default hyperparameters for model_name
#     """
#     # TODO: Actually implement this for all subclasses.
#     model_wrapper_class = _resolve_model(model_name)
#     return model_wrapper_class.default_config(model_name)


# def _get_data_format(train_data):
#     """Return data format ("torch", "numpy", or "dir") for the data."""

#     # TODO: Maybe also check val_data and test_data.
#     if isinstance(train_data, torch.utils.data.Dataset):
#         return "torch"
#     elif isinstance(train_data, list) and isinstance(train_data[0], np.ndarray):
#         return "numpy"
#     elif isinstance(train_data, str) or isinstance(train_data, Path):
#         return "dir"
#     else:
#         raise ValueError(
#             "Could not recognize format of train_data and/or test_data. Supported "
#             "formats: list of numpy arrays, torch dataset"
#         )
