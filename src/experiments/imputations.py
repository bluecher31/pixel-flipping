import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass
from pathlib import Path
import pickle

from src.config.helpers import compare_cfg, select_folder
from src.explainers.helper_images import load_segmentor
from src.datasets.imagenet import unnormalize_image
from src.experiments.resources import load_data

from conditional_explainer.imputers.abstract_imputer import Imputer
from src.interface import Image
from src.config.config import ImputationsConfig, SegmentationConfig

from typing import List, Tuple
from numpy.typing import NDArray
from omegaconf import DictConfig


@dataclass
class Imputation:
    imputation: NDArray
    n_occluded_superpixels: int
    image_name: str
    imputer_name: str


def get_segmentation(cfg_segmentation: SegmentationConfig, image: Image, n_occluded_superpixels: int) -> NDArray:
    """Randomly segment 'n_occluded_superpixels' patches for the provided image. """
    generate_superpixel = load_segmentor(cfg_segmentation)
    segmentation = generate_superpixel(image)

    superpixels = np.unique(segmentation)
    # ensure that always the same superpixels are occluded
    rng = np.random.default_rng(0)
    rng.shuffle(superpixels)  # get fixed random ordering

    occluded_superpixels = superpixels[:n_occluded_superpixels]
    for s in occluded_superpixels:
        mask = segmentation == s
        segmentation[mask] *= -1
    return segmentation


def get_imputation(segmentation: NDArray, image: Image, imputer: Imputer) \
        -> Imputation:
    """Returns a single imputation."""
    imputation_raw = imputer.impute(data=image.image[None], segmentation_coalitions=segmentation[None, None],
                                    n_imputations=1)
    imputation = Imputation(imputation=np.squeeze(imputation_raw),
                            n_occluded_superpixels=int(np.sum(np.unique(segmentation) < 0)),
                            image_name=image.image_name, imputer_name=imputer.imputer_name
                            )
    return imputation


def store_imputations(imputations: List[Imputation]):
    """
    Store all imputation individually wrt. number of imputed superpixels
    params.root_experiment/samples/imputations/imputer/n_superpixel/compactness_slic/image_name/5.pickle
    """

    for imputation in imputations:
        file_path = Path.cwd() / imputation.image_name / f'{imputation.n_occluded_superpixels}.pickle'
        file_path.parent.mkdir(exist_ok=True, parents=True)
        with open(file_path, 'wb') as file:
            pickle.dump(imputation, file)


def root_imputations(cfg: ImputationsConfig, root: Path = Path.cwd().parent) -> Path:
    return root / cfg.dataset.name / 'imputations' / cfg.imputer.name


def load_imputations(cfg: ImputationsConfig, config_folders: List[Tuple[Path, DictConfig]]) -> [List[Image], List[List[Imputation]]]:
    """
    Returns a list imputations corresponding to different images.
    """

    def compare_cfg_imputations(cfg_base, cfg_test) -> bool:
        keys = ['dataset', 'imputer', 'segmentation']
        return compare_cfg(cfg_base, cfg_test, keys=keys, set_keys=['platform', 'dataset.root_data'])

    folders = [path for (path, cfg_test) in config_folders if compare_cfg_imputations(cfg, cfg_test)]

    folder_imputations = select_folder(folders, rule='most_subfolders')

    all_images = [dir_sample for dir_sample in folder_imputations.glob('*')
                  if dir_sample.is_dir() and dir_sample.name != '.hydra']
    all_images.sort()

    list_imputations = []
    for image_path in all_images:
        # search for all available imputations
        all_path_imputations = [path_imputation for path_imputation in image_path.glob('*')]
        all_path_imputations.sort(key=lambda path: int(path.name.strip('.pickle')))

        imputations = [pickle.load(open(file_path, 'rb')) for file_path in all_path_imputations]
        list_imputations.append(imputations)

    images = load_data(dataset=cfg.dataset, n_samples=len(list_imputations),
                       train=False, format_numpy=False)
    images_sorted = sorted(images, key=lambda image: image.image_name)
    list_imputations_sorted = sorted(list_imputations,
                                     key=lambda imputations: imputations[0].image_name)

    assert len(images_sorted) == len(list_imputations_sorted), 'Could not find imputations.'
    if 'n_samples' in cfg:
        images_sorted, list_imputations_sorted = \
            images_sorted[:cfg.n_samples], list_imputations_sorted[:cfg.n_samples]

    return images_sorted, list_imputations_sorted


def extract_fixed_n_imputations(list_imputations: List[List[Imputation]], n_occluded_superpixels: int) -> List[Imputation]:
    """Returns imputations with 'n_occluded_superpixels' occluded superpixel for each image. """
    assert len(list_imputations) > 0, 'No imputations.'
    imputations_fixed_n = []
    for imputations in list_imputations:
        for imp in imputations:
            if imp.n_occluded_superpixels == n_occluded_superpixels:
                imputations_fixed_n.append(imp)
                break
    assert len(list_imputations) == len(imputations_fixed_n), f'Did not find an imputation for each image.\n' \
                                                              f'len(list_imputations) = {len(list_imputations)}'
    return imputations_fixed_n


def plot_imagenet(ax: plt.Axes, image_np: NDArray):
    """Visualize a single image from imagenet."""
    image_np = unnormalize_image(image_np=image_np)
    ax.imshow(np.transpose(image_np, (1, 2, 0)), cmap=plt.cm.binary)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)


def plot_imputations(image: Image, imputations: List[Imputation]):
    """Plots imputations from a single image with different fractions of occlusion."""
    #   raw image - few imputations -> many imputations
    n_cols = len(imputations) + 1
    imputer_name = imputations[0].imputer_name

    title = f'{imputer_name} - {image.image_name}'
    fig = plt.figure(title, figsize=(2.5 * n_cols, 2))
    ax = fig.add_subplot(1, n_cols, 1)

    ax.set_title(f'{imputer_name}')
    plot_imagenet(ax=ax, image_np=image.image)

    for index, imputation in enumerate(imputations):
        assert image.image_name == imputation.image_name
        assert imputer_name == imputation.imputer_name
        ax = fig.add_subplot(1, n_cols, index + 2)
        ax.set_title(f'$s$ = {imputation.n_occluded_superpixels}')
        plot_imagenet(ax=ax, image_np=imputation.imputation)

    plt.tight_layout(pad=0.1)
