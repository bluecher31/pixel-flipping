import torch
import numpy as np
import os

from torchvision import datasets
import torchvision.transforms as transforms

from src.interface import Image
from typing import List, Union
from numpy.typing import NDArray
from pathlib import Path
import torchvision

imagenet_norm = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class UnNormalize(torchvision.transforms.Normalize):
    """Revers normalization."""
    def __init__(self, mean, std, *args, **kwargs):
        new_mean = [-m / s for m, s in zip(mean, std)]
        new_std = [1 / s for s in std]
        super().__init__(new_mean, new_std, *args, **kwargs)


def normalize_image(image_np: NDArray) -> NDArray:
    """Wrapper to create imagenet Image instance with human comprehensible normalization."""
    normalize = transforms.Normalize(**imagenet_norm)
    image_human_view_np = normalize.forward(torch.tensor(image_np)).numpy()
    return image_human_view_np


def unnormalize_image(image_np: NDArray) -> NDArray:
    """Wrapper to create imagenet Image instance with model comprehensible normalization."""
    unnormalize = UnNormalize(**imagenet_norm)
    image_human_view_np = unnormalize.forward(torch.tensor(image_np)).numpy()
    return np.clip(image_human_view_np, 0, 1)


def preprocess_imagenet(image: Image) -> NDArray:
    image_new = unnormalize_image(image_np=image.image)
    return np.transpose(image_new, (1, 2, 0))


def load_imagenet_data(root: str, n: int, train: bool, format_numpy: bool, image_size: int) \
        -> Union[List[Image], NDArray]:
    """Returns sample whith rgb values [0, 1]. """
    root_abspath = Path(os.path.abspath(root))

    assert root_abspath.exists(), f'Root path to imagenet dataset does not exist.\n {root_abspath}'
    val_dataset = datasets.ImageFolder(str(root_abspath),
                                       transforms.Compose([
                                           # old
                                           # transforms.Resize(232),
                                           # transforms.CenterCrop(224),
                                           # new
                                           transforms.Resize(image_size),
                                           transforms.CenterCrop(image_size),
                                           # transforms.PILToTensor(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(**imagenet_norm)
                                       ]))

    if train is False:
        rng = np.random.default_rng(0)
    else:
        rng = np.random.default_rng(1)          # ensure different samples for imputation

    n_val_images = len(val_dataset)
    permutation = np.arange(0, n_val_images)
    rng.shuffle(permutation)  # fixed random permutation

    if format_numpy is False:  # convert into custom Image format
        list_images = []
        for i, index in enumerate(permutation[:n]):
            img_tensor, label = val_dataset[index]
            image_np = np.array(img_tensor)

            image_path, _ = val_dataset.samples[index]
            file_name = Path(image_path).name
            image_name = file_name.strip('.JPEG')

            class_name = val_dataset.classes[label]

            im = Image(image=image_np, label=int(label), class_name=class_name,
                       image_name=image_name, dataset='imagenet')

            list_images.append(im)
        images = list_images
    else:  # return numpy format
        list_images = []
        for i, index in enumerate(permutation[:n]):
            img_tensor, label = val_dataset[index]
            image_np = np.array(img_tensor)
            list_images.append(image_np)
        images = np.stack(list_images)

    return images


def load_class_names(fname: Path = Path('.').absolute().parent / f'src/datasets' / 'imagenet_classes.txt') -> NDArray:
    """Loads all 1000 class names for imagenet."""
    assert fname.exists()
    imagenet_classes = np.loadtxt(fname=fname.as_posix(), dtype=str, delimiter=';')
    assert len(imagenet_classes) == 1000
    return imagenet_classes
