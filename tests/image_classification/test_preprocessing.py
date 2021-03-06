import comet_ml  # noqa: F401
import pytest
import numpy as np
import torch

from conftest import create_dataset, create_image

from traintool.image_classification.preprocessing import (
    recognize_data_format,
    torch_to_numpy,
    numpy_to_torch,
    files_to_numpy,
    files_to_torch,
    load_image,
    recognize_image_format,
    get_num_classes,
)


@pytest.fixture
def numpy_data():
    return create_dataset(data_format="numpy", seed=0, grayscale=False)


@pytest.fixture
def torch_data():
    return create_dataset(data_format="torch", seed=0, grayscale=False)


@pytest.fixture
def files_data(tmp_path):
    return create_dataset(data_format="files", seed=0, tmp_path=tmp_path)


@pytest.fixture
def numpy_image():
    return create_image(data_format="numpy", seed=0, grayscale=False)


@pytest.fixture
def torch_image():
    return create_image(data_format="torch", seed=0, grayscale=False)


@pytest.fixture
def files_image(tmp_path):
    return create_image(data_format="files", seed=0, tmp_path=tmp_path)


def test_recognize_data_format(numpy_data, torch_data, files_data):
    
    # correct data formats
    assert recognize_data_format(numpy_data) == "numpy"
    assert recognize_data_format(torch_data) == "pytorch-dataset"
    assert recognize_data_format(files_data) == "files"

    # incorrect data formats
    with pytest.raises(ValueError):
        recognize_data_format(None)
    with pytest.raises(ValueError):
        recognize_data_format([1, 2, 3])
    with pytest.raises(FileNotFoundError):
        recognize_data_format("non/existent/dir/123")


def test_recognize_image_format(numpy_image, torch_image, files_image):

    # correct image formats
    assert recognize_image_format(numpy_image) == "numpy"
    assert recognize_image_format(files_image) == "files"

    # incorrect image formats
    with pytest.raises(ValueError):
        recognize_image_format(None)
    with pytest.raises(ValueError):
        recognize_image_format([1, 2, 3])
    with pytest.raises(FileNotFoundError):
        recognize_image_format("non/existent/file/123")


def test_torch_to_numpy(numpy_data, torch_data):
    converted_data = torch_to_numpy(torch_data)
    assert np.allclose(converted_data[0], numpy_data[0])
    assert np.allclose(converted_data[1], numpy_data[1])


def test_numpy_to_torch(numpy_data, torch_data):
    converted_data = numpy_to_torch(numpy_data)
    # Note that we compare with tolerance of 0.1 here, because due to conversion to PIL,
    # values are not exactly preserved.
    assert torch.allclose(converted_data[0][0], torch_data[0][0], atol=0.1)
    assert converted_data[0][1] == torch_data[0][1]

    resized_converted_data = numpy_to_torch(
        numpy_data, resize=256, crop=224, mean=[0.1, 0.1, 0.1], std=[0.1, 0.1, 0.1]
    )
    assert resized_converted_data[0][0].shape[1] == 224
    assert resized_converted_data[0][0].shape[2] == 224


def test_files_to_numpy(files_data, numpy_data):
    converted_data = files_to_numpy(files_data)
    assert converted_data[0][0].shape == numpy_data[0][0].shape

    resized_converted_data = files_to_numpy(
        files_data, resize=256, crop=224, mean=[0.1, 0.1, 0.1], std=[0.1, 0.1, 0.1]
    )
    assert resized_converted_data[0][0].shape[1] == 224
    assert resized_converted_data[0][0].shape[2] == 224


def test_files_to_torch(files_data, torch_data):
    converted_data = files_to_torch(files_data)
    assert converted_data[0][0].shape == torch_data[0][0].shape

    resized_converted_data = files_to_numpy(
        files_data, resize=256, crop=224, mean=[0.1, 0.1, 0.1], std=[0.1, 0.1, 0.1]
    )
    assert resized_converted_data[0][0].shape[1] == 224
    assert resized_converted_data[0][0].shape[2] == 224


# TODO: Maybe add tests for to_numpy and to_torch, but note that these are kinda
#   redundant to the tests above.


def test_load_image(tmp_path):
    data = create_dataset(grayscale=False, data_format="files", tmp_path=tmp_path)

    # Select a random image.
    image_path = next(data.rglob("*.png"))

    # torch
    img = load_image(image_path, resize=50, crop=40)
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 40, 40)

    # numpy
    img = load_image(image_path, resize=50, crop=40, to_numpy=True)
    assert isinstance(img, np.ndarray)
    assert img.shape == (3, 40, 40)


def test_get_num_classes(numpy_data, files_data):
    assert get_num_classes(numpy_data) == 4
    assert get_num_classes(files_data) == 4
