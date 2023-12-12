import numpy as np
import os

from torchvision import datasets

from src.interface import Image
from typing import List, Union
from numpy.typing import NDArray


def load_mnist_data(root: str, n: int, train: bool, format_numpy: bool) -> Union[List[Image], NDArray]:
    """Always returns the same random samples which are suitable for the corresponding 'Model'. """
    root_abspath = os.path.abspath(root)
    ds = datasets.MNIST(root=root_abspath, download=True, train=train)
    samples = np.array(ds.data/255)
    labels = ds.targets

    if format_numpy is False:  # convert into custom Image format
        list_images = []
        for i in range(n):
            image_np = samples[i]
            label = labels[i]
            class_name = ds.classes[label].split(' - ')[1]   # returns 'zero'/'one/'two/...
            # define a unique image_name since loaded from compressed file
            image_name = f'mnist_{"test" if train is False else "train"}_{i:06}'

            im = Image(image=image_np, label=int(label), class_name=class_name,
                       image_name=image_name, dataset='mnist')
            list_images.append(im)
        images = list_images
    else:  # return numpy format
        images = samples[:n]

    return images


if __name__ == '__main__':
    print('Run src/datasets/mnist.py as main. ')
    # _ = load_mnist_data(n=10, root='./data')      # change root pass
