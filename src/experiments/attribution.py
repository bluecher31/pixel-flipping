import numpy as np
import pickle
from pathlib import Path

import scipy.stats

from src.explainers.helper_images import generate_superpixel
from src.config.helpers import compare_cfg, find_all_existing_path, select_folder
from src.experiments.resources import load_data

from typing import List, Tuple
from numpy.typing import NDArray
from omegaconf import DictConfig
from src.interface import Image, Attribution


def save_attributions(attributions: List[Attribution]):
    """
    attribution: /dataset/model/explainer/property1/property2/.../image_key/attribution.pickle
    image: /dataset/test_samples/image_key.pickle
    """
    for attr in attributions:
        dir_attr = Path.cwd() / attr.image_name
        dir_attr.mkdir(exist_ok=True, parents=True)
        with open(dir_attr / f'attributions.pickle', 'wb') as file:
            pickle.dump(attr, file)


def root_attribution(cfg, root: Path = Path.cwd().parent / 'outputs') -> Path:
    return root / cfg.dataset.name / cfg.model.name / 'attributions' / cfg.explainer.name


def generate_random_attributions(cfg, images: List[Image]) -> List[Attribution]:
    attributions = []
    for image in images:
        n_channels = 3 if cfg.dataset.name == 'imagenet' else 1
        random_values = np.random.randn(cfg.dataset.image_size ** 2 * n_channels)
        heatmap = np.squeeze(random_values.reshape((n_channels, cfg.dataset.image_size, cfg.dataset.image_size)))
        attribution = Attribution(
            heatmap=heatmap,
            dataset=cfg.dataset.name,
            image_name=image.image_name,
            explainer='random',
            prediction=None,
            model=cfg.model.name,
            label=None,
            explainer_properties={},
        )
        attributions.append(attribution)
    return attributions


def load_attributions(cfg, config_folders: List[Tuple[Path, DictConfig]], index_folder: int = 0) -> [List[Image], List[Attribution]]:
    """Retrieves all pairs of attribution/images associated with a specific set of params."""
    images = load_data(dataset=cfg.dataset, n_samples=cfg.n_samples,
                       train=False, format_numpy=False)
    images_sorted = sorted(images, key=lambda image: image.image_name)

    if cfg.explainer.name == 'random':
        attributions = generate_random_attributions(cfg, images_sorted)
    else:
        def compare_cfg_attributions(cfg_base, cfg_test) -> bool:
            keys = ['dataset', 'model', 'explainer']
            return compare_cfg(cfg_base, cfg_test, keys=keys, set_keys=['platform'])

        folders = [path for (path, cfg_test) in config_folders if compare_cfg_attributions(cfg, cfg_test)]
        assert len(folders) > 0, f'Please enter a least a single folder.\n{cfg}'
        folder = select_folder(folders, rule='most_subfolders', index=index_folder)

        path_to_attributions = find_all_existing_path(folder=folder, file_path='attributions.pickle')
        attributions = [pickle.load(open(file, 'rb')) for file in path_to_attributions]
        assert len(attributions) > 0, f'No attributions found for cfg: \n{cfg}'

    if len(attributions) > cfg.n_samples:         # filter the correct samples
        images_names = [image.image_name for image in images]
        attributions = [attr for attr in attributions if attr.image_name in images_names]

    attributions_sorted = sorted(attributions, key=lambda attr: attr.image_name)

    if len(attributions) < cfg.n_samples:          # invalid, will fail later
        print(f'Found n = {len(attributions)} attributions but expected {cfg.n_samples}:\n'
              f'folder = {folder}\n'
              f'{cfg}')
    else:
        for image, attr in zip(images_sorted, attributions_sorted):
            assert image.image_name == attr.image_name

    assert len(images) == len(attributions), \
        f'Not equal number of images ({len(images)}) and attributions ({len(attributions)})'
    return images_sorted, attributions_sorted


def convert_to_superpixel_attributions(heatmap: NDArray, n_superpixel: int, compactness_slic: float) -> List[float]:
    """Pool heatmap into attributions per superpixel."""
    segmentation = generate_superpixel(image=heatmap, n_segments=n_superpixel, compactness=compactness_slic)
    attribution_per_superpixel = []
    for s in np.unique(segmentation):
        mask = segmentation == s
        attribution = np.mean(heatmap[mask])
        attribution_per_superpixel.append(float(attribution))
    return attribution_per_superpixel


def cosine_similarity(arr1: NDArray, arr2: NDArray) -> float:
    """Calculate cosine similarity between two arrays (heatmaps)."""
    a = arr1.flatten()
    b = arr2.flatten()
    a_times_b = a * b
    cos_sim = a_times_b.sum(axis=-1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1)
    return cos_sim


def calculate_similarity(attributions1: List[Attribution], attributions2: List[Attribution], which: str) \
        -> [float, float]:
    """Calculate the average similarity between two different attributions using the cosine similarity."""
    # print([a1.image_name == a2.image_name for a1, a2 in zip(attributions1, attributions2)])
    assert len(attributions1) == len(attributions2), 'Please provide an equal number of attributions.'
    similarities = []
    for a1, a2 in zip(attributions1, attributions2):
        assert a1.image_name == a2.image_name, f'{a1.image_name} != {a2.image_name}'
        if which == 'cosine similarity':
            similarity = cosine_similarity(arr1=a1.heatmap, arr2=a2.heatmap)
        elif which == 'spearman':
            corr = scipy.stats.spearmanr(np.stack([a1.heatmap, a2.heatmap]).reshape(2, -1), axis=1)
            similarity = corr.correlation
        elif which == 'pearson':
            similarity, _ = scipy.stats.pearsonr(x=a1.heatmap.flatten(), y=a2.heatmap.flatten())
        elif which == 'kendalltau':
            n_superpixel = a1.explainer_properties['n_superpixel']
            compactness = a1.explainer_properties['compactness_slic']
            attr1 = convert_to_superpixel_attributions(heatmap=a1.heatmap, n_superpixel=n_superpixel,
                                                                     compactness_slic=compactness)
            attr2 = convert_to_superpixel_attributions(heatmap=a2.heatmap, n_superpixel=n_superpixel,
                                                       compactness_slic=compactness)
            # corr = scipy.stats.kendalltau(x=attr1, y=attr2)
            corr = scipy.stats.kendalltau(x=a1.heatmap.flatten(), y=a2.heatmap.flatten())
            similarity = corr.correlation
        else:
            raise ValueError(f'This type of correlation/similarity is not defined: which = {which}.')
        similarities.append(similarity)
    assert len(similarities) > 0, 'Need at least a single measurement.'
    mean = float(np.mean(similarities))
    error = float(np.std(similarities)/np.sqrt(len(similarities)))

    return mean, error
