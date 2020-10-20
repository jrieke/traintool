import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset
from torchvision import datasets, transforms
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

    # Check for pytorch dataset.
    if isinstance(data, Dataset):
        return "pytorch-dataset"

    # Check for iterable of numpy arrays (images, labels).
    try:
        if (
            len(data) == 2
            and isinstance(data[0], np.ndarray)
            and isinstance(data[1], np.ndarray)
        ):
            # Check for correct shape of images.
            images, labels = data
            if len(images.shape) == 4 and (images.shape[1] in [1, 3]):
                return "numpy"
            else:
                raise ValueError(
                    "Shape of images not understood, should be "
                    "num_samples x color_channels (1 or 3) x height x width, "
                    f"is: {images.shape}"
                )
    except TypeError:  # data is no iterable
        pass

    # Check for path to directory.
    try:
        if Path(data).exists():
            return "files"
        else:
            raise FileNotFoundError(f"Data directory does not exist: {data}")
    except TypeError:  # not a file or dir
        pass

    # If all checks failed...
    raise ValueError(
        "Data format not recognized. Supported formats: list of numpy "
        "arrays, torch dataset, directory of image files"
    )

    # TODO: Maybe add pytorch tensors.


def to_torch(data) -> Union[Dataset, None]:
    """Convert data from any format to torch datasets."""

    # Handle empty dataset.
    if data is None:
        return None

    # Recognize data format and convert accordingly.
    data_format = recognize_data_format(data)
    if data_format == "pytorch-dataset":
        return data
    elif data_format == "numpy":
        return numpy_to_torch(data)
    elif data_format == "files":
        return files_to_torch(data)
    else:
        raise RuntimeError()


def to_numpy(data) -> Union[List[np.ndarray], None]:
    """Convert data from any format to lists of numpy arrays [input, target]."""
    # Handle empty dataset.
    if data is None:
        return None

    # Recognize data format and convert accordingly.
    data_format = recognize_data_format(data)
    if data_format == "pytorch-dataset":
        return torch_to_numpy(data)
    elif data_format == "numpy":
        return data
    elif data_format == "files":
        return files_to_numpy(data)
    else:
        raise RuntimeError()


class ImageDataset(Dataset):
    """
    Generic dataset for images that can deal with PIL images, numpy arrays, 
    or torch tensors.
    """

    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def numpy_to_torch(
    data: List[np.ndarray],
    resize: int = None,
    crop: int = None,
    mean: List = None,
    std: List = None,
) -> Dataset:
    """Convert data from list of numpy arrays [input, target] to torch datasets."""
    # Unpack arrays.
    images, labels = data

    # Rescale images to 0-255 and convert to uint8.
    # TODO: Right now this is done for each dataset individually, which is probably
    #   not a big deal, because all datasets should usually contain values from the
    #   complete range of values. Ideally though, this should probably be done based
    #   on the values of the train dataset.
    images = (images - np.min(images)) / np.ptp(images) * 255
    images = images.astype(np.uint8)

    # If images are grayscale, convert to RGB by duplicating channels.
    if images.shape[1] == 1:
        images = np.stack((images[:, 0],) * 3, axis=1)

    # Convert to channels last format (required for transforms.ToPILImage).
    images = channels_last(images)

    # Set up transform to convert to PIL image, do manipulations, and convert to tensor.
    # TODO: Converting to PIL and then to tensor is not super efficient, find a better
    #   method.
    transform = create_transform(
        from_numpy=True, resize=resize, crop=crop, mean=mean, std=std
    )

    # [0.485, 0.456, 0.406]
    # [0.229, 0.224, 0.225]

    # Convert labels to tensors.
    labels = torch.from_numpy(labels).long()

    # Construct dataset.
    dataset = ImageDataset(images, labels, transform=transform)
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


def files_to_numpy(
    root: Path,
    resize: int = None,
    crop: int = None,
    mean: List = None,
    std: List = None,
) -> List[np.ndarray]:
    """Load image files into pytorch dataset and convert to numpy array."""
    dataset = files_to_torch(root, resize=resize, crop=crop, mean=mean, std=std)
    numpy_data = torch_to_numpy(dataset)
    return numpy_data


def files_to_torch(
    root: Path,
    resize: int = None,
    crop: int = None,
    mean: List = None,
    std: List = None,
) -> Dataset:
    """Load image files into pytorch dataset."""
    # Set up transform for loading and converting files.
    # TODO: For the sklearn models, this probably shouldn't load in size 224,
    #   especially in case the images are smaller.
    transform = create_transform(resize=resize, crop=crop, mean=mean, std=std)

    # Load images from folder.
    dataset = datasets.ImageFolder(root, transform=transform)
    return dataset


def create_transform(
    from_numpy: bool = False,
    resize: int = None,
    crop: int = None,
    mean: List = None,
    std: List = None,
):
    """Creates a torchvision transform to convert images from PIL or numpy."""
    t = []
    if from_numpy:
        t.append(transforms.ToPILImage())
    if resize is not None:
        t.append(transforms.Resize(resize))
    if crop is not None:
        t.append(transforms.CenterCrop(crop))
    t.append(transforms.ToTensor())
    if mean is not None and std is not None:
        t.append(transforms.Normalize(mean=mean, std=std),)
    return transforms.Compose(t)
