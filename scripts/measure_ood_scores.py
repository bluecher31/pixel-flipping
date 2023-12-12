import torch

import numpy as np

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import logging
from tqdm import tqdm

import pickle
from pathlib import Path

from src.explainers.helper_images import get_idx_filtered_n_superpixel
from src.config.helpers import save_config, resolve_imagenet, load_special_config
from src.experiments.resources import load_model, load_data, load_imputer
from src.experiments.imputations import get_imputation, get_segmentation

from conditional_explainer.imputers.abstract_imputer import Imputer
from src.config.config import MeasureOODScoresConfig, SegmentationConfig
from src.interface import Image, Model

from numpy.typing import NDArray
from typing import List


def root_ood(dataset_name: str, model_name: str, imputer_name: str) -> Path:
    return Path.cwd().parent / dataset_name / model_name / 'ood' / imputer_name


def get_occluded_fraction(segmentation: NDArray) -> float:
    mask = segmentation < 0
    return np.sum(mask) / mask.size


def get_occluded_predictions(cfg_segmentation: SegmentationConfig,
                             images: List[Image], imputer: Imputer, model: Model, occluded_superpixel: int) \
        -> [NDArray, List[float]]:
    segmentations = [get_segmentation(cfg_segmentation, image, occluded_superpixel)
                     for image in images]
    actual_fraction_occluded = [get_occluded_fraction(seg) for seg in segmentations]
    imputations = [get_imputation(segmentation, image, imputer)
                   for image, segmentation in zip(images, segmentations)]
    imputations_np = np.stack([imp.imputation for imp in imputations])
    if imputer.imputer_name != 'internal':
        occluded_predictions = model.predict_proba(imputations_np)  # np, shape: (n_samples, n_classes)
    else:   # don't use imptuations
        assert model.model_name == 'Madry_ViT', 'Only implemented for the madry ViT.'
        occluded_predictions_list = [model.occluded_model_fn(image.image, segmentation)
                                     for image, segmentation in zip(images, segmentations)]
        occluded_predictions = np.stack(occluded_predictions_list)
    return occluded_predictions, actual_fraction_occluded


def store_occluded_predictions(n_superpixel: int, occluded_prediction, actual_fraction_occluded,
                               images: List[Image], cardinality: int):
    """dataset/model/imputer/n_superpixel/S=[]/image_key/occluded_predictions.pickle"""

    # print(f'Storing images/attributions at root directory: \n{root_abspath}')
    for image, predictions, fraction in zip(images, occluded_prediction, actual_fraction_occluded):
        dir_predictions = Path.cwd() / f'n_superpixel={n_superpixel}, s={cardinality}' / image.image_name
        dir_predictions.mkdir(exist_ok=True, parents=True)
        with open(dir_predictions / 'occluded_prediction.pickle', 'wb') as file:
            pickle.dump(predictions, file)
        with open(dir_predictions / 'actual_fraction_occluded.pickle', 'wb') as file:
            pickle.dump(fraction, file)


cs = ConfigStore.instance()
cs.store(name="config", node=MeasureOODScoresConfig)
OmegaConf.register_new_resolver('resolve_imagenet', resolve_imagenet)


@hydra.main(config_path='../src/conf', config_name='measure_ood_scores', version_base=None)
def main(cfg: MeasureOODScoresConfig):
    logging.info(msg='\n' + OmegaConf.to_yaml(cfg))
    save_config(cfg, Path.cwd())

    if cfg.filter_images is True:
        cfg_segement_anything = load_special_config('segment_anything', type='segmentation')
        cfg_segement_anything.n_superpixel = cfg.segmentation.n_superpixel
        idx_images = get_idx_filtered_n_superpixel(cfg_segmentation=cfg_segement_anything)
        test_images = load_data(dataset=cfg.dataset, n_samples=cfg.n_samples, idx_images=idx_images)
    else:
        test_images = load_data(dataset=cfg.dataset, n_samples=cfg.n_samples)

    model = load_model(model_cfg=cfg.model, dataset_name=cfg.dataset.name)

    imputer = load_imputer(imputer_cfg=cfg.imputer, dataset=cfg.dataset)

    for cardinality in tqdm(np.linspace(1, cfg.segmentation.n_superpixel, 10, dtype=int)):
        occluded_predictions, actual_fraction_occluded = get_occluded_predictions(
            cfg_segmentation=cfg.segmentation, images=test_images, imputer=imputer, model=model,
            occluded_superpixel=cardinality
        )
        store_occluded_predictions(n_superpixel=cfg.segmentation.n_superpixel,
                                   occluded_prediction=occluded_predictions,
                                   actual_fraction_occluded=actual_fraction_occluded,
                                   images=test_images,
                                   cardinality=cardinality)


if __name__ == '__main__':
    main()
