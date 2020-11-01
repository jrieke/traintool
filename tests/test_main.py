import comet_ml
import pytest
from tensorboardX import SummaryWriter

from conftest import create_dataset

from traintool.main import (
    connect_comet,
    train,
    load,
    # default_config,
    _resolve_model,
    # _update_config,
    _write_info_file,
    _read_info_file,
    _create_comet_experiment,
    _create_tensorboard_writer,
)
from traintool.model_wrapper import ModelWrapper
from traintool.utils import DummyExperiment


def test_resolve_model():
    assert issubclass(_resolve_model("resnet18"), ModelWrapper)
    with pytest.raises(ValueError):
        _resolve_model("non-existing-model-123")


# def test_update_config():
#     default_config = {"param1": 123, "param2": "hello"}
#     config = {"param2": "hello again"}
#     invalid_config = {"param3": "this doesn't exist"}

#     updated_config = _update_config(default_config, config)
#     assert isinstance(updated_config, dict)
#     assert len(updated_config.keys()) == 2
#     assert updated_config["param1"] == 123
#     assert updated_config["param2"] == "hello again"
#     with pytest.raises(ValueError):
#         _update_config(default_config, invalid_config)


def test_write_read_info_file(tmp_path):
    _write_info_file(tmp_path, "some-model", {"param1": 123})

    assert (tmp_path / "info.yml").exists()

    content = _read_info_file(tmp_path)
    assert content["model_name"] == "some-model"
    assert content["config"]["param1"] == 123


def test_create_comet_experiment():
    # Fake experiment object (traintool.connect_comet not called)
    experiment = _create_comet_experiment(save=True)
    experiment.log_metric("fake-metric", 1)
    assert isinstance(experiment, DummyExperiment)

    # Real experiment object
    connect_comet("fake-api-key")  # won't connect with with dry_run == True
    experiment = _create_comet_experiment(save=True)
    experiment.log_metric("fake-metric", 1)
    assert isinstance(experiment, comet_ml.Experiment)


def test_create_tensorboard_writer(tmp_path):
    writer = _create_tensorboard_writer(tmp_path)
    assert isinstance(writer, SummaryWriter)
    writer.add_scalar("param", 123)
    writer.close()


def test_train(tmp_path):
    train_data = create_dataset()

    with pytest.raises(ValueError):
        train("non-existing-model-123", None)

    # with pytest.raises(ValueError):
    #     train("random-forest", None, config={"non-existing-parameter": 123})

    # With save=False (this has to be checked first, so tmp_path is still empty)
    model_wrapper = train(
        "random-forest",
        train_data=train_data,
        dry_run=True,
        save=False,
    )
    assert isinstance(model_wrapper, ModelWrapper)
    assert not any(tmp_path.iterdir())  # is empty dir

    # With all datasets
    model_wrapper = train(
        "random-forest",
        train_data=train_data,
        val_data=train_data,
        test_data=train_data,
        dry_run=True,
        save=tmp_path,
    )
    assert isinstance(model_wrapper, ModelWrapper)
    assert (tmp_path / "info.yml").exists()
    assert (tmp_path / "model.joblib").exists()
    
    # With only train data
    model_wrapper = train(
        "random-forest",
        train_data=train_data,
        dry_run=True,
        save=tmp_path,
    )
    assert isinstance(model_wrapper, ModelWrapper)
    assert (tmp_path / "info.yml").exists()
    assert (tmp_path / "model.joblib").exists()


def test_load(tmp_path):
    train_data = create_dataset()
    train(
        "random-forest",
        train_data=train_data,
        test_data=train_data,
        dry_run=True,
        save=tmp_path,
    )
    loaded_model_wrapper = load(tmp_path)
    assert isinstance(loaded_model_wrapper, ModelWrapper)
    assert loaded_model_wrapper.model is not None
    # TODO: Make some more checks on loaded_model_wrapper, possibly using a dummy 
    #   wrapper class.


# def test_default_config():
#     assert isinstance(default_config("simple-cnn"), dict)
#     assert "lr" in default_config("simple-cnn")
#     assert "n_estimators" in default_config("random-forest")
#     with pytest.raises(ValueError):
#         default_config("non-existing-model-123")
