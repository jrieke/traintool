import sys
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset
from typing import Union, List
from pathlib import Path
import imageio


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# TODO: Maybe build generator around this which returns all the possible data formats
#   that are allowed.
def create_dataset(
    data_format: str = "numpy",
    size: int = 28,
    grayscale: bool = True,
    num_samples: int = 4,
    seed: int = 0,
    tmp_path: Path = None,
) -> Union[List[np.ndarray], TensorDataset, Path]:
    """
    Create fake data for image classification, including images (values from 0 to 1) 
    and labels (10 classes). 

    Args:
        data_format (str, optional): "numpy" (for list of numpy arrays 
            [images, labels]) or "torch" (for torch.utils.data.Dataset) or "files" 
            (for directory of image files). Defaults to "numpy".
        size (int, optional): Width and height of the images. Defaults to 28.
        grayscale (bool, optional): If True, images are grayscale and of size 28x28, 
            otherwise they have 3 color channels and are 224x224. Defaults to False.
        num_samples (int, optional): Number of samples in the dataset. Defaults to 4.
        seed (int, optional): Seed for data generation. Defaults to 0.

    Returns:
        Union[List[np.ndarray], torch.utils.data.Dataset, Path]: The generated fake 
            data. 
    """

    # Create raw data (seeded, so always the same data).
    np.random.seed(seed)
    num_channels = 1 if grayscale else 3
    # Images are in channels first format.
    images = np.random.rand(num_samples, num_channels, size, size)
    labels = np.random.randint(10, size=num_samples)

    # Convert to correct output format.
    if data_format == "numpy":
        return [images, labels]
    elif data_format == "torch":
        dataset = TensorDataset(
            torch.from_numpy(images).float(), torch.from_numpy(labels).long(),
        )
        return dataset
    elif data_format == "files":
        if tmp_path is None:
            raise ValueError("tmp_path must be given if data_format is files")
        for i, (image, label) in enumerate(zip(images, labels)):
            image = image.transpose((1, 2, 0))
            image_dir = tmp_path / "data" / f"{label}"
            image_dir.mkdir(parents=True, exist_ok=True)
            imageio.imwrite(image_dir / f"{i}.png", image)
        return tmp_path / "data"
    else:
        raise ValueError(f"data_format not supported: {data_format}")


def create_image(data_format: str = "numpy", **kwargs):
    """Creates a single image, using the same keyword args as create_dataset."""
    data = create_dataset(data_format=data_format, **kwargs)
    # TODO: Maybe implement this without relying on create_dataset
    #   to make it more efficient.

    if data_format == "numpy" or data_format == "torch":
        return data[0][0]
    elif data_format == "files":
        return next(data.rglob("*.png"))
    else:
        raise RuntimeError()
