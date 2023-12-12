import torch
import numpy as np
import pandas as pd
import pickle
from dataclasses import dataclass
from pathlib import Path

from conditional_explainer.imputers.abstract_imputer import Imputer
from src.explainers.helper_images import load_segmentor

from src.config.config import PixelFlippingConfig
from src.config.helpers import select_folder, compare_cfg

from omegaconf import OmegaConf

from typing import List, Tuple
from numpy.typing import NDArray
from omegaconf import DictConfig
from src.experiments.attribution import Attribution
from src.interface import Image, Model


def root_pixel_flipping(dataset_name: str, model_name: str, imputer_name_pf: str, explainer_name: str) -> Path:
    return Path.cwd().parent / dataset_name / model_name / 'pixel_flipping' / imputer_name_pf / explainer_name


@dataclass
class PixelFlipping:
    """Stores results from any pixel flipping experiment."""
    percentage_deleted: NDArray
    occluded_prediction: NDArray
    image_name: str


def convert_heatmap_to_superpixels(heatmap: NDArray, segmentation: NDArray[int]) -> pd.DataFrame:
    """
    Summarizes the pixel-wise heatmap into the mean values per superpixels.

    Args:
        heatmap: pixel-wise attributions, shape: image.image.shape
        segmentation: positive integers segmenting the image into superpixels, max
        (segmentation) = n_superpixel, min(segmentation) = 1, shape: image.image.shape

    Returns:
        attribution mean per superpixel, shape: (n_superpixels)
    """
    assert np.min(segmentation) == 1, 'Superpixel segmentation needs to start with 1.'
    assert heatmap.shape == segmentation.shape, f'Not matching shapes.\n segmentation.shape={segmentation.shape}'
    df_attributions = pd.DataFrame(np.unique(segmentation), columns=['segment_index'])
    df_attributions['attribution'] = None       # init empty attributions

    for i, feature_index in enumerate(df_attributions['segment_index']):
        mask = segmentation == feature_index
        attributions = heatmap[mask]
        df_attributions.loc[i, 'attribution'] = np.mean(attributions)
    return df_attributions


def select_label(predictions: NDArray, label: int, which: str) -> NDArray:
    if which == 'label':
        target_probability = predictions[..., label]
        # if segmentation_occluded.min() > 0 and np.argmax(occluded_prediction) != image.label:
        #     print(f'Model predicts the wrong label. \n'
        #           f'max: {np.max(occluded_prediction):.3f}\n'
        #           f'label: {occluded_prediction[image.label]:.3f}.'
        #           )

    elif which == 'max':
        target_probability = np.max(predictions, -1)
    else:
        raise ValueError(f'This target value is not defined. which: {which}.')
    return target_probability


def _occluded_model(model: Model, imputer: Imputer, image: Image, segmentation_occluded: NDArray[int], n_imputations: int,
                    which: str = 'label') \
        -> float:
    """
    Args:
        image: current sample
        mask: which regions should be marginalized via imputations -> probably change to segmentation_coalitions
        n_imputations: numerical fidelity of imputation integral

    Returns:
        occluded prediction for the provided sample and the original correct image class.
    """
    imputations = imputer.impute(image.image[None], segmentation_occluded[None, None], n_imputations)
    # shape: (n_imputations, 1, 1, *input_shape, *shape_x)

    if imputer.imputer_name != 'internal':
        occluded_predictions = model.predict_proba(np.squeeze(imputations))
        # shape: (n_imputations, n_classes)
        target_probabilities = select_label(occluded_predictions, label=image.label, which=which)
        target_probability = np.mean(target_probabilities, axis=0)
    else:
        assert model.model_name == 'Madry_ViT', 'Only implemented for the madry ViT.'
        occluded_prediction = model.occluded_model_fn(image.image, segmentation_occluded)
        target_probability = select_label(occluded_prediction, label=image.label, which=which)

    return float(target_probability)


def _consistency_check(attribution: Attribution, cfg: PixelFlippingConfig):
    """Checks whether provided attribution are compatible with the current PixelFlippingExperiment."""
    if 'segmentation' in attribution.explainer_properties:
        dict_segmentation = attribution.explainer_properties['segmentation']
        assert cfg.explainer.segmentation == dict_segmentation, \
            f'Superpixel-based attribution methods require matching values for compactness_slic. \n' \
            f'{cfg.explainer.segmentation} =! {dict_segmentation}'


def _generate_occlusion_spacing(n_measurments: int, n_superpixels: int, scale_spacing: str = 'linear') -> NDArray:
    """returns shape: (n_measurements,), dtype = int"""
    if scale_spacing == 'linear':
        superixpels_deleted = np.linspace(1, n_superpixels, n_measurments, dtype=int)
    elif scale_spacing == 'log':
        superixpels_deleted = np.geomspace(1, n_superpixels, n_measurments, dtype=int)
    elif scale_spacing == 'log_inverse':        # high resolution for gradually presenting high-relevant features
        superixpels_deleted = n_superpixels - np.geomspace(1, n_superpixels, n_measurments, dtype=int)
    else:
        raise ValueError(f'Invalid argument: scale_spacing = {scale_spacing}.')

    superixpels_deleted_np = np.unique(np.concatenate(([0, n_superpixels], superixpels_deleted)))
    return np.sort(superixpels_deleted_np)


def pixel_flipping(cfg: PixelFlippingConfig, image: Image, attribution: Attribution,
                   imputer: Imputer, model: Model) -> PixelFlipping:
    """
    Check documentation here:
    https://github.com/understandable-machine-intelligence-lab/Quantus

    """
    assert image.image_name == attribution.image_name
    _consistency_check(attribution=attribution, cfg=cfg)

    generate_patches = load_segmentor(cfg_segmentation=cfg.segmentation_pf)
    segmentation = generate_patches(image)

    superpixels_deleted = _generate_occlusion_spacing(n_superpixels=cfg.segmentation_pf.n_superpixel,
                                                      n_measurments=cfg.n_measurements,
                                                      scale_spacing=cfg.scale_spacing)

    df_attributions = convert_heatmap_to_superpixels(heatmap=attribution.heatmap, segmentation=segmentation)
    df_attributions = df_attributions.sort_values(by=['attribution'])

    occluded_predictions = np.zeros_like(superpixels_deleted, dtype=float)
    actual_percentage_deleted = np.zeros_like(superpixels_deleted, dtype=float)
    for index, n_occluded_superpixel in enumerate(superpixels_deleted):
        segmentation_occluded = segmentation.copy()

        if n_occluded_superpixel == 0:
            pass    # keep all segments unchanged
        else:
            occluded_superpixels = list(df_attributions['segment_index'][-n_occluded_superpixel:]) if cfg.most_relevant_first else \
                list(df_attributions['segment_index'][:n_occluded_superpixel])     # select superpixels with minimal relevance

            for superpixel in occluded_superpixels:
                mask = segmentation_occluded == superpixel
                assert np.sum(mask) > 0, f'Empty mask, nothing to occlude. {n_occluded_superpixel}'
                segmentation_occluded[mask] *= -1

        # fill mask according to attribution, n_occluded_superpixel/n_superpixel
        occluded_prediction = _occluded_model(model=model, imputer=imputer, image=image,
                                              segmentation_occluded=segmentation_occluded,
                                              n_imputations=cfg.n_imputations_pf, which=cfg.which_target)
        occluded_predictions[index] = occluded_prediction
        actual_percentage_deleted[index] = float(np.sum(segmentation_occluded < 0) / segmentation.size)

    pf_result = PixelFlipping(percentage_deleted=actual_percentage_deleted,
                              occluded_prediction=occluded_predictions, image_name=image.image_name)

    return pf_result


def store_pixel_flipping(pf_result: PixelFlipping):
    """
    Stores experiments according to 'pf_MostRF_imputer.pickle'/'pf_LeastRF_imputer.pickle' (MostRF: Most Relevant First)
    Folder identical to attribution.
    Files are not overwritten but appended with an integer index. (Same name can correspond to different params.)

    Args:
        pf_results: list of pixel flipping experiments
        overwrite: set to True if you want to keep previous calculations.
    """
    dir_pf = Path().cwd() / pf_result.image_name
    dir_pf.mkdir(exist_ok=True, parents=True)
    file_name = f'result.pickle'

    path = dir_pf / file_name

    with open(path, 'wb') as file:
        pickle.dump(pf_result, file)


def compare_cfg_pf(cfg_base: DictConfig, cfg_test: DictConfig) -> bool:
    keys = ['dataset', 'model', 'imputer_pf', 'scale_spacing', 'n_measurements', 'explainer', 'segmentation_pf',
            'most_relevant_first', 'which_target']
    if cfg_test.explainer.name == 'random':
        cfg_test.explainer = OmegaConf.create({"name": 'random'})

    if ('which_target' in cfg_test) is False:         # add key to cfg_test
        cfg_test['which_target'] = 'label'

    if compare_cfg(cfg_base, cfg_test, keys=keys, set_keys=['platform']) is False:
        return False
    return True


def load_pixel_flipping(cfg: PixelFlippingConfig, config_folders: List[Tuple[Path, DictConfig]]) -> List[PixelFlipping]:
    folders = [path for (path, cfg_test) in config_folders if compare_cfg_pf(cfg, cfg_test)]
    assert len(folders) > 0, f'Missing measurements for {cfg.explainer}.\n{cfg}'
    folder = select_folder(folders, rule='most_subfolders')

    files_measurements = [dir_image / 'result.pickle' for dir_image in folder.glob('*') if dir_image.is_dir()]
    files_measurements_sorted = sorted([file for file in files_measurements if file.exists()])
    assert len(files_measurements_sorted) > 0, f'No files found for imputer: {cfg}.\n{folder}'

    pf_results = [pickle.load(open(file, 'rb')) for file in files_measurements_sorted]
    return pf_results
