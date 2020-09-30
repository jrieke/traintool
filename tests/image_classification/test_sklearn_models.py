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
    wrapper = SklearnImageClassificationWrapper("random-forest")
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
    wrapper = SklearnImageClassificationWrapper("random-forest")
    wrapper._create_model({"n_estimators": 10})
    assert isinstance(wrapper.model, RandomForestClassifier)
    assert wrapper.model.n_estimators == 10


@pytest.mark.parametrize(
    "data_format", ["numpy", "torch"],
)
def test_train(data_format, tmp_path):
    data = create_image_classification_data(output_format=data_format, grayscale=True)
    wrapper = SklearnImageClassificationWrapper("random-forest")

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

    assert isinstance(wrapper.model, RandomForestClassifier)
    assert wrapper.scaler is not None
    assert (tmp_path / "model.joblib").exists()


def test_load(wrapper):
    loaded_wrapper = SklearnImageClassificationWrapper.load(
        wrapper.out_dir, "random-forest"
    )
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
