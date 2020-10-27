import comet_ml
import pytest
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from tensorboardX import SummaryWriter

from conftest import create_dataset, create_image

from traintool.image_classification import SklearnImageClassificationWrapper
from traintool.utils import DummyExperiment


# TODO: Maybe persist this on module level to save time.
@pytest.fixture
def wrapper(tmp_path):
    """A simple wrapper around random-forest model"""
    data = create_dataset(grayscale=False)
    wrapper = SklearnImageClassificationWrapper("random-forest", {}, tmp_path)
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
    wrapper = SklearnImageClassificationWrapper(
        "random-forest", {"n_estimators": 10}, tmp_path
    )
    wrapper._create_model()
    assert isinstance(wrapper.model, RandomForestClassifier)
    assert wrapper.model.n_estimators == 10


@pytest.mark.parametrize("data_format", ["numpy", "files"])
@pytest.mark.parametrize("grayscale", [True, False])
def test_train(data_format, grayscale, tmp_path):
    data = create_dataset(
        data_format=data_format, grayscale=grayscale, size=28, tmp_path=tmp_path,
    )
    wrapper = SklearnImageClassificationWrapper("random-forest", {}, tmp_path)

    wrapper._train(
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
    loaded_wrapper._load()
    assert isinstance(loaded_wrapper.model, RandomForestClassifier)
    assert loaded_wrapper.scaler is not None


@pytest.mark.parametrize("data_format", ["numpy", "files"])
def test_predict(wrapper, data_format, tmp_path):
    image = create_image(grayscale=False, data_format=data_format, tmp_path=tmp_path)

    result = wrapper.predict(image)
    assert "predicted_class" in result
    assert "probabilities" in result
    print(result)

    assert isinstance(result["predicted_class"], int)
    assert isinstance(result["probabilities"], np.ndarray)
    # TODO: Assert that length of probabilities is equal to the number of classes.
    # TODO: Assert that predicted_class is below the number of classes.


def test_raw(wrapper):
    raw = wrapper.raw()
    assert "model" in raw
    assert "scaler" in raw
