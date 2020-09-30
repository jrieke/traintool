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


# @pytest.mark.parametrize(
#     "model_id,data_format",
#     [
#         ("simple-cnn", "numpy"),
#         ("simple-cnn", "torch"),
#         ("resnet18", "numpy"),
#         ("resnet18", "torch"),
#     ],
# )
# def test_train(model_id, data_format, config, tmp_path):
#     """
#     Test SimpleCnnWrapper.train_and_save by checking that the method runs without errors
#     (does not control if the model actually learns!).
#     """
#     # TODO: Maybe refactor to one method for all models

#     # Create temp output dir.
#     out_dir = tmp_path / data_format
#     out_dir.mkdir(parents=True, exist_ok=True)

#     # Create data.
#     data = create_image_classification_data(
#         output_format=data_format, grayscale=(model_id == "simple-cnn")
#     )

#     # Call train_and_save method.
#     wrapper = TorchImageClassificationWrapper(model_id)
#     wrapper.train(
#         train_data=data,
#         val_data=data,
#         test_data=data,
#         config=config,
#         experiment=DummyExperiment(),
#         out_dir=out_dir,
#         dry_run=False,
#     )


# @pytest.mark.parametrize(
#     "model_id,stride", [("simple-cnn", (1, 1)), ("resnet18", (2, 2))]
# )
# def test_create_model(model_id, stride):
#     """Test SimpleCnnWrapper.create_model by checking stride of first conv layer."""
#     wrapper = TorchImageClassificationWrapper(model_id)
#     wrapper._create_model()
#     # Check stride of first conv layer. This is just a random check. Cannot compare
#     # models directly because weights differ at each initialization.
#     assert wrapper.model.conv1.stride == stride

# TODO: Maybe persist this on module level to save time.
# @pytest.fixture
# def wrapper(tmp_path):
#     """A simple wrapper around random-forest model"""
#     data = create_image_classification_data(grayscale=True)
#     wrapper = SklearnImageClassificationWrapper("random-forest")
#     wrapper.train(
#         train_data=data,
#         val_data=None,
#         test_data=None,
#         config={},
#         writer=SummaryWriter(write_to_disk=False),
#         experiment=DummyExperiment(),
#         out_dir=tmp_path,
#         dry_run=True,
#     )
#     return wrapper


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
    adadelta_lr = wrapper._create_optimizer({"optimizer": "adadelta", "lr": 123}, params)
    assert isinstance(adadelta, optim.Adadelta)
    assert adadelta_lr.defaults["lr"] == 123
    
    # unknown optimizer
    with pytest.raises(ValueError):
        wrapper._create_optimizer({"optimizer": "unknown-optimizer123"}, params)
    


@pytest.mark.parametrize("data_format", ["numpy", "torch"])
def test_train(data_format, tmp_path):
    data = create_image_classification_data(output_format=data_format, grayscale=True)
    wrapper = TorchImageClassificationWrapper("simple-cnn")

    wrapper.train(
        train_data=data,
        val_data=data,
        test_data=data,
        config={},
        writer=SummaryWriter(write_to_disk=False),
        experiment=DummyExperiment(),
        out_dir=tmp_path,
        dry_run=True,
    )

    assert isinstance(wrapper.model, nn.Module)
    assert (tmp_path / "model.pt").exists()


# def test_load(wrapper):
#     loaded_wrapper = SklearnImageClassificationWrapper.load(
#         wrapper.out_dir, "random-forest"
#     )
#     assert isinstance(loaded_wrapper.model, RandomForestClassifier)
#     assert loaded_wrapper.scaler is not None


# def test_predict(wrapper):
#     data = create_image_classification_data(grayscale=True)
#     result = wrapper.predict(data[0][0:1])
#     assert "predicted_class" in result
#     assert "probabilities" in result


# def test_raw(wrapper):
#     raw = wrapper.raw()
#     assert "model" in raw
#     assert "scaler" in raw
