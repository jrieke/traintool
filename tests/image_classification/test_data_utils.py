import comet_ml
import pytest
from typing import List
import numpy as np
import torch

from conftest import create_image_classification_data

from traintool.image_classification.data_utils import (
    recognize_data_format,
    torch_to_numpy,
    numpy_to_torch,
)


@pytest.fixture
def numpy_data():
    return create_image_classification_data(output_format="numpy", seed=0)


@pytest.fixture
def torch_data():
    return create_image_classification_data(output_format="torch", seed=0)


@pytest.fixture
def files_data(tmp_path):
    # TODO: Create some images in the dir.
    return tmp_path


def test_recognize_data_format(numpy_data, torch_data, files_data):
    assert recognize_data_format(numpy_data) == "numpy"
    assert recognize_data_format(torch_data) == "pytorch-dataset"
    assert recognize_data_format(files_data) == "files"
    
    with pytest.raises(ValueError):
        recognize_data_format(None)
        
    with pytest.raises(ValueError):
        recognize_data_format([1, 2, 3])
    
    with pytest.raises(FileNotFoundError):
        recognize_data_format("non/existent/dir/123")

def test_torch_to_numpy(numpy_data, torch_data):
    converted_data = torch_to_numpy(torch_data)
    assert np.allclose(converted_data[0], numpy_data[0])
    assert np.allclose(converted_data[1], numpy_data[1])


def test_numpy_to_torch(numpy_data, torch_data):
    converted_data = numpy_to_torch(numpy_data)
    assert torch.allclose(converted_data[0][0], torch_data[0][0])
    assert converted_data[0][1] == torch_data[0][1]


# TODO: Write tests for to_torch and to_numpy.
