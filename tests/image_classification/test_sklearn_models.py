import comet_ml
import pytest
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from tensorboardX import SummaryWriter

from conftest import create_image_classification_data

from traintool.image_classification import SklearnImageClassificationWrapper
from traintool.utils import DummyExperiment


# TODO: Maybe persist this on module level to save time.
@pytest.fixture
def wrapper(tmp_path):
    """A simple wrapper around random-forest model"""
    data = create_image_classification_data(grayscale=True)
    wrapper = SklearnImageClassificationWrapper("random-forest", {}, tmp_path)
    wrapper.train(
        train_data=data,
        val_data=None,
        test_data=None,
        writer=SummaryWriter(write_to_disk=False),
        experiment=DummyExperiment(),
        dry_run=True,
    )
    return wrapper


def test_create_model(tmp_path):
    wrapper = SklearnImageClassificationWrapper(
        "random-forest", {"n_estimators": 10}, tmp_path
    )
    wrapper._create_model()
    assert isinstance(wrapper.model, RandomForestClassifier)
    assert wrapper.model.n_estimators == 10


@pytest.mark.parametrize("data_format", ["numpy", "files"])
@pytest.mark.parametrize("grayscale", [True, False])
def test_train(data_format, grayscale, tmp_path):
    data = create_image_classification_data(
        data_format=data_format, grayscale=grayscale, size=28, tmp_path=tmp_path,
    )
    wrapper = SklearnImageClassificationWrapper("random-forest", {}, tmp_path)

    wrapper.train(
        train_data=data,
        val_data=data,
        test_data=data,
        writer=SummaryWriter(write_to_disk=False),
        experiment=DummyExperiment(),
        dry_run=True,
    )

    assert isinstance(wrapper.model, RandomForestClassifier)
    assert wrapper.scaler is not None
    assert (tmp_path / "model.joblib").exists()


def test_load(wrapper):
    loaded_wrapper = SklearnImageClassificationWrapper(
        "random-forest", {}, wrapper.out_dir
    )
    loaded_wrapper.load()
    assert isinstance(loaded_wrapper.model, RandomForestClassifier)
    assert loaded_wrapper.scaler is not None


def test_predict(wrapper):
    data = create_image_classification_data(grayscale=True)
    result = wrapper.predict(data[0][0:1])
    assert "predicted_class" in result
    assert "probabilities" in result


def test_raw(wrapper):
    raw = wrapper.raw()
    assert "model" in raw
    assert "scaler" in raw
