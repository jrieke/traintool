import comet_ml
import pytest
from typing import List
import numpy as np
import torch

from conftest import create_image_classification_data

from traintool.image_classification import data_utils


@pytest.fixture
def numpy_data():
    return create_image_classification_data(output_format="numpy", seed=0)


@pytest.fixture
def torch_data():
    return create_image_classification_data(output_format="torch", seed=0)


def test_torch_to_numpy(numpy_data, torch_data):
    converted_data = data_utils.torch_to_numpy(torch_data)
    assert np.allclose(converted_data[0], numpy_data[0])
    assert np.allclose(converted_data[1], numpy_data[1])


def test_numpy_to_torch(numpy_data, torch_data):
    converted_data = data_utils.numpy_to_torch(numpy_data)
    assert torch.allclose(converted_data[0][0], torch_data[0][0])
    assert converted_data[0][1] == torch_data[0][1]


# TODO: Write tests for to_torch and to_numpy.
