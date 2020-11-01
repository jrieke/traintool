from comet_ml import Experiment
from typing import Union, Type
from pathlib import Path
import yaml
from tensorboardX import SummaryWriter
import tempfile
import editdistance
import numpy as np

from . import utils
from . import image_classification
from .model_wrapper import ModelWrapper


comet_config = {}
# TODO: Add a method to change the project dir.
project_dir = Path.cwd() / "traintool-experiments"


model_dict = {
    "simple-cnn": image_classification.TorchImageClassificationWrapper,
    "resnet18": image_classification.TorchImageClassificationWrapper,
    "alexnet": image_classification.TorchImageClassificationWrapper,
    "vgg16": image_classification.TorchImageClassificationWrapper,
    "squeezenet": image_classification.TorchImageClassificationWrapper,
    "densenet": image_classification.TorchImageClassificationWrapper,
    "inception": image_classification.TorchImageClassificationWrapper,
    "googlenet": image_classification.TorchImageClassificationWrapper,
    "shufflenet": image_classification.TorchImageClassificationWrapper,
    "mobilenet": image_classification.TorchImageClassificationWrapper,
    "resnext50_32x4d": image_classification.TorchImageClassificationWrapper,
    "wide_resnet50_2": image_classification.TorchImageClassificationWrapper,
    "mnasnet": image_classification.TorchImageClassificationWrapper,
    "random-forest": image_classification.SklearnImageClassificationWrapper,
    "gradient-boosting": image_classification.SklearnImageClassificationWrapper,
    "gaussian-process": image_classification.SklearnImageClassificationWrapper,
    "logistic-regression": image_classification.SklearnImageClassificationWrapper,
    "sgd": image_classification.SklearnImageClassificationWrapper,
    "perceptron": image_classification.SklearnImageClassificationWrapper,
    "passive-aggressive": image_classification.SklearnImageClassificationWrapper,
    "gaussian-nb": image_classification.SklearnImageClassificationWrapper,
    "k-neighbors": image_classification.SklearnImageClassificationWrapper,
    "mlp": image_classification.SklearnImageClassificationWrapper,
    "svc": image_classification.SklearnImageClassificationWrapper,
    "linear-svc": image_classification.SklearnImageClassificationWrapper,
    "decision-tree": image_classification.SklearnImageClassificationWrapper,
    "extra-tree": image_classification.SklearnImageClassificationWrapper,
}


def _resolve_model(model_name: str) -> Type[ModelWrapper]:
    """Return class of model wrapper that is used for model_name."""
    if model_name in model_dict:
        return model_dict[model_name]
    else:
        # Find most similar model name.
        models = list(model_dict.keys())
        distances = [editdistance.eval(model_name, model) for model in models]
        most_similar = models[np.argmin(distances)]

        raise ValueError(
            f"Model not recognized: {model_name} " f"(did you mean {most_similar}?)"
        )


# def _update_config(default_config: dict, config: dict) -> dict:
#     """Update values in default_config with values in config"""
#     final_config = default_config.copy()
#     for key, value in config.items():
#         if key not in final_config:
#             raise ValueError(
#                 "config contains a parameter that is not supported for this model: "
#                 f"{key}"
#             )
#         else:
#             final_config[key] = value
#     return final_config


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


def connect_comet(api_key: str = None, project_name: str = None) -> None:
    """Connect comet.ml account to traintool (required to track metrics)."""
    comet_config["api_key"] = api_key
    comet_config["project_name"] = project_name


def train(
    model_name: str,
    train_data,
    val_data=None,
    test_data=None,
    config: dict = None,
    save: Union[bool, str, Path] = True,
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
        save (Union[bool, str, Path], optional): Whether to store model and logs to disk
            (and to comet.ml if connected). If False, nothing will be saved. If True, 
            artefacts will be saved to a timestamped directory in 
            ./traintool-experiments. If a directory (str or Path) is given, artefacts 
            will be saved there. Defaults to True. 
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
        if save is True:  # timestamped dir in ./traintool-experiments
            out_dir = project_dir / experiment_name
            out_dir.mkdir(parents=True, exist_ok=False)
        elif save is False:  # temporary dir
            # TODO: For convenience, we are just making out_dir a temporary directory
            #   here, which will be discarded. Instead, we shouldn't save anything at
            #   all because it might be more performant.
            out_dir = Path(tmp_dir)
        else:  # use save as dir
            out_dir = Path(save)
            out_dir.mkdir(parents=True, exist_ok=True)
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
