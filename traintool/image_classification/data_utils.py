import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset
from typing import List, Tuple


def to_torch(data) -> Dataset:
    """Convert data from any format to torch datasets."""
    if isinstance(data, Dataset):
        return data
    elif isinstance(data, list) and isinstance(data[0], np.ndarray):
        return numpy_to_torch(data)
    else:
        raise ValueError(
            "Could not recognize data format. Supported formats: list of numpy arrays "
            "[images, labels], torch dataset"
        )


def to_numpy(data) -> List[np.ndarray]:
    """Convert data from any format to lists of numpy arrays [input, target]."""
    if isinstance(data, Dataset):
        return torch_to_numpy(data)
    elif isinstance(data, list) and isinstance(data[0], np.ndarray):
        return data
    else:
        raise ValueError(
            "Could not recognize data format. Supported formats: list of numpy arrays "
            "[images, labels], torch dataset"
        )


def numpy_to_torch(data: List[np.ndarray],) -> Dataset:
    """Convert data from list of numpy arrays [input, target] to torch datasets."""

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

