from pickle import load
import comet_ml
import pytest
import numpy as np
import torchvision
import torch
from pathlib import Path
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from conftest import create_dataset, create_image

from traintool.image_classification.torch_models import (
    TorchImageClassificationWrapper,
    SimpleCnn,
)
from traintool.utils import DummyExperiment


# TODO: Maybe persist this on module level to save time.
@pytest.fixture
def wrapper(tmp_path):
    """A simple wrapper around random-forest model"""
    data = create_dataset(data_format="numpy", grayscale=False, size=224)
    wrapper = TorchImageClassificationWrapper("resnet18", {}, tmp_path)
    wrapper._train(
        train_data=data,
        val_data=None,
        test_data=None,
        writer=SummaryWriter(write_to_disk=False),
        experiment=DummyExperiment(),
        dry_run=True,
    )
    return wrapper


def test_create_model(tmp_path):

    # resnet18
    wrapper = TorchImageClassificationWrapper("resnet18", {}, tmp_path)
    wrapper._create_model()
    assert isinstance(wrapper.model, nn.Module)
    assert wrapper.model.fc.out_features == 1000
    # pretrained=True is not tested here because it takes too long

    # resnet18 with custom num_classes
    wrapper = TorchImageClassificationWrapper("resnet18", {"num_classes": 10}, tmp_path)
    wrapper._create_model()
    assert isinstance(wrapper.model, nn.Module)
    assert wrapper.model.fc.out_features == 10

    # simple-cnn
    wrapper = TorchImageClassificationWrapper("simple-cnn", {}, tmp_path)
    wrapper._create_model()
    assert isinstance(wrapper.model, SimpleCnn)


# TODO: Test torch datasets as well.
# @pytest.mark.parametrize("data_format", ["numpy", "files"])
# @pytest.mark.parametrize("grayscale", [True, False])
# def test_preprocess_for_training(data_format, grayscale, tmp_path):
#     data = create_dataset(
#         data_format=data_format, grayscale=grayscale, tmp_path=tmp_path,
#     )
#     wrapper = TorchImageClassificationWrapper("resnet18")
#     batch_size = 2
#     loaders = wrapper._preprocess_for_training(
#         data, data, data, config={"batch_size": batch_size}
#     )

#     for loader in loaders:
#         assert isinstance(loader, DataLoader)
#         images, labels = next(iter(loader))
#         assert len(images) == batch_size
#         assert len(labels) == batch_size
#         assert len(labels.shape) == 1
#         assert len(images.shape) == 4
#         assert images.shape[1] == 3
#         assert images.shape[2] == 224
#         assert images.shape[3] == 224


def test_create_optimizer(tmp_path):
    wrapper = TorchImageClassificationWrapper("resnet18", {}, tmp_path)
    wrapper._create_model()

    # default
    default_optimizer = wrapper._create_optimizer()
    assert isinstance(default_optimizer, optim.Optimizer)

    # adadelta
    wrapper.config["optimizer"] = "adadelta"
    adadelta = wrapper._create_optimizer()
    assert isinstance(adadelta, optim.Adadelta)

    # adadelta with lr
    wrapper.config["optimizer"] = "adadelta"
    wrapper.config["lr"] = 123
    adadelta_lr = wrapper._create_optimizer()
    assert isinstance(adadelta, optim.Adadelta)
    assert adadelta_lr.defaults["lr"] == 123

    # unknown optimizer
    wrapper.config["optimizer"] = "unknown-optimizer123"
    with pytest.raises(ValueError):
        wrapper._create_optimizer()


# TODO: Test torch datasets.
@pytest.mark.parametrize("data_format", ["numpy", "files"])
@pytest.mark.parametrize("grayscale", [True, False])
def test_train(data_format, grayscale, tmp_path):
    # TODO: Test for grayscale = False and different size.
    # data = create_dataset(
    #     data_format=data_format, size=28, grayscale=True
    # )
    # wrapper = TorchImageClassificationWrapper("simple-cnn")
    data = create_dataset(
        data_format=data_format, grayscale=grayscale, tmp_path=tmp_path
    )
    wrapper = TorchImageClassificationWrapper("resnet18", {}, tmp_path)

    # TODO: Test both resnet18 and simple-cnn with a few different configurations of data.
    # TODO: Test with and without val/test data.
    # TODO: Check that something was written to writer and experiment.
    wrapper._train(
        train_data=data,
        val_data=data,
        test_data=data,
        writer=SummaryWriter(write_to_disk=False),
        experiment=DummyExperiment(),
        dry_run=True,  # True,
    )

    assert isinstance(wrapper.model, nn.Module)
    assert (tmp_path / "model.pt").exists()


def test_load(wrapper):
    loaded_wrapper = TorchImageClassificationWrapper("resnet18", {}, wrapper.out_dir)
    loaded_wrapper._load()
    assert isinstance(loaded_wrapper.model, nn.Module)
    # TODO: Maybe do some more tests here.


@pytest.mark.parametrize("data_format", ["numpy", "files"])
def test_predict(wrapper, data_format, tmp_path):
    image = create_image(grayscale=False, data_format=data_format, tmp_path=tmp_path)

    result = wrapper.predict(image)
    assert "predicted_class" in result
    assert "probabilities" in result
    print(result)

    assert isinstance(result["predicted_class"], int)
    assert isinstance(result["probabilities"], np.ndarray)
    assert result["probabilities"].ndim == 1
    # TODO: Assert that length of probabilities is equal to the number of classes.
    # TODO: Assert that predicted_class is below the number of classes.


def test_raw(wrapper):
    raw = wrapper.raw()
    assert "model" in raw
    assert isinstance(raw["model"], nn.Module)
