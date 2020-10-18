import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset
from typing import List, Tuple, Union
from pathlib import Path


def channels_first(images: np.ndarray) -> np.ndarray:
    """Convert images from channels last to channels first format."""
    return images.transpose((0, 3, 1, 2))


def channels_last(images: np.ndarray) -> np.ndarray:
    """Convert images from channels first to channels last format."""
    return images.transpose((0, 2, 3, 1))


def recognize_data_format(data) -> str:
    """Returns a string which describes the data format of `data`."""
    if isinstance(data, Dataset):
        return "pytorch-dataset"

    try:
        if (
            len(data) == 2
            and isinstance(data[0], np.ndarray)
            and isinstance(data[1], np.ndarray)
        ):
            return "numpy"
    except TypeError:  # no iterable
        pass

    try:
        if Path(data).exists():
            return "files"
        else:
            raise FileNotFoundError(f"Data directory not found: {data}")
    except TypeError:  # not a file or dir
        pass

    raise ValueError(
        "Data format not recognized. Supported formats: list of numpy "
        "arrays, torch dataset, directory of image files"
    )

    # TODO: Maybe add pytorch tensors.


def to_torch(data) -> Dataset:
    """Convert data from any format to torch datasets."""
    data_format = recognize_data_format(data)
    if data_format == "pytorch-dataset":
        return data
    elif data_format == "numpy":
        return numpy_to_torch(data)


def to_numpy(data) -> List[np.ndarray]:
    """Convert data from any format to lists of numpy arrays [input, target]."""
    data_format = recognize_data_format(data)
    if data_format == "pytorch-dataset":
        return torch_to_numpy(data)
    elif data_format == "numpy":
        return data


def numpy_to_torch(data: List[np.ndarray]) -> Union[Dataset, None]:
    """Convert data from list of numpy arrays [input, target] to torch datasets."""

    # Handle empty dataset.
    if data is None:
        return None

    # Unpack arrays.
    images, labels = data

    # Convert arrays to torch tensors and wrap in TensorDataset.
    dataset = TensorDataset(
        torch.from_numpy(images).float(), torch.from_numpy(labels).long(),
    )
    return dataset


def torch_to_numpy(data: Dataset) -> List[np.ndarray]:
    """Convert data from torch dataset to list of numpy arrays [input, target]."""

    # Create empty numpy arrays.
    images_shape = (len(data), *data[0][0].shape)
    images = np.zeros(images_shape)
    labels = np.zeros(len(data))

    # Fill arrays with samples from torch dataset.
    # Note that samples in torch datasets may change from iteration to iteration
    # because of random transforms.
    # TODO: What to do if data is too large for memory?
    for i, (image, label) in enumerate(data):
        images[i] = image
        labels[i] = label
    return [images, labels]

