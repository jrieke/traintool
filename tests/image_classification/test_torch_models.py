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

from conftest import create_image_classification_data

from traintool.image_classification.torch_models import (
    TorchImageClassificationWrapper,
    SimpleCnn,
)
from traintool.utils import DummyExperiment


# TODO: Maybe persist this on module level to save time.
@pytest.fixture
def wrapper(tmp_path):
    """A simple wrapper around random-forest model"""
    data = create_image_classification_data(
        output_format="torch", grayscale=False, size=224
    )
    wrapper = TorchImageClassificationWrapper("resnet18")
    wrapper.train(
        train_data=data,
        val_data=None,
        test_data=None,
        config={},
        writer=SummaryWriter(write_to_disk=False),
        experiment=DummyExperiment(),
        out_dir=tmp_path,
        dry_run=True,
    )
    return wrapper


def test_create_model():
    wrapper = TorchImageClassificationWrapper("resnet18")
    wrapper._create_model({})
    assert isinstance(wrapper.model, nn.Module)
    # pretrained=True is not tested here because it takes too long

    wrapper = TorchImageClassificationWrapper("simple-cnn")
    wrapper._create_model({})
    assert isinstance(wrapper.model, SimpleCnn)


@pytest.mark.parametrize("data_format", ["numpy", "torch"])
def test_preprocess(data_format):
    data = create_image_classification_data(output_format=data_format)
    wrapper = TorchImageClassificationWrapper("resnet18")
    batch_size = 2
    data_loader = wrapper._preprocess(data, {"batch_size": batch_size})
    assert isinstance(data_loader, DataLoader)
    batch_images, batch_labels = next(iter(data_loader))
    assert len(batch_images) == batch_size
    assert len(batch_labels) == batch_size
    # TODO: Check shape of images once they are processed correctly.


def test_create_optimizer():
    wrapper = TorchImageClassificationWrapper("resnet18")
    params = [nn.Parameter(torch.zeros(1))]

    # default
    default_optimizer = wrapper._create_optimizer({}, params)
    assert isinstance(default_optimizer, optim.Optimizer)

    # adadelta
    adadelta = wrapper._create_optimizer({"optimizer": "adadelta"}, params)
    assert isinstance(adadelta, optim.Adadelta)

    # adadelta with lr
    adadelta_lr = wrapper._create_optimizer(
        {"optimizer": "adadelta", "lr": 123}, params
    )
    assert isinstance(adadelta, optim.Adadelta)
    assert adadelta_lr.defaults["lr"] == 123

    # unknown optimizer
    with pytest.raises(ValueError):
        wrapper._create_optimizer({"optimizer": "unknown-optimizer123"}, params)


@pytest.mark.parametrize("data_format", ["numpy", "torch"])
def test_train(data_format, tmp_path):
    # TODO: Test for grayscale = False and different size.
    # data = create_image_classification_data(
    #     output_format=data_format, size=28, grayscale=True
    # )
    # wrapper = TorchImageClassificationWrapper("simple-cnn")
    data = create_image_classification_data(
        output_format=data_format, size=28, grayscale=True
    )
    wrapper = TorchImageClassificationWrapper("simple-cnn")

    # TODO: Test both resnet18 and simple-cnn with a few different configurations of data.
    # TODO: Test with and without val/test data.
    # TODO: Check that something was written to writer and experiment.
    wrapper.train(
        train_data=data,
        val_data=data,
        test_data=data,
        config={},
        writer=SummaryWriter(write_to_disk=False),
        experiment=DummyExperiment(),
        out_dir=tmp_path,
        dry_run=True,  # True,
    )

    assert isinstance(wrapper.model, nn.Module)
    assert (tmp_path / "model.pt").exists()


def test_load(wrapper):
    loaded_wrapper = TorchImageClassificationWrapper.load(wrapper.out_dir, "resnet18")
    assert isinstance(loaded_wrapper.model, nn.Module)
    # TODO: Maybe do some more tests here.


# TOOD: Enable this once the preprocessing is done properly.
# def test_predict(wrapper):
#     data = create_image_classification_data(grayscale=True)
#     result = wrapper.predict(data[0][0:1])
#     assert "predicted_class" in result
#     assert "probabilities" in result


def test_raw(wrapper):
    raw = wrapper.raw()
    assert "model" in raw
    assert isinstance(raw["model"], nn.Module)
