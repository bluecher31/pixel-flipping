import copy

import numpy as np
import torch

from hydra.utils import get_original_cwd

from conditional_explainer.imputers.simple_imputers import TrainSetImputer, ConstantValueImputer, IdentityImputer
from conditional_explainer.imputers.color_sampling_imputer import ColorHistogramImputer

from src.explainers.conditional_explainer import PredDiff, ShapleyValues, KernelSHAP

from src.datasets.mnist import load_mnist_data

from src.datasets.imagenet import load_imagenet_data, unnormalize_image, normalize_image
from src.models.imagenet import load_imagenet_model

from conditional_explainer.imputers.abstract_imputer import Imputer
from src.interface import Model, ImageExplainer, Image, ConfigExplainer
from src.config.config import DatasetConfig, ImputerConfig, ModelConfig
from typing import List, Union
from numpy.typing import NDArray

from pathlib import Path


def load_data(dataset: DatasetConfig, n_samples: int, train: bool = False, format_numpy: bool = False,
              idx_images: List[int] = None) -> Union[List[Image], NDArray]:
    if dataset.name == 'mnist':
        images = load_mnist_data(root=dataset.root_data, n=n_samples, train=train, format_numpy=format_numpy)
    elif dataset.name == 'imagenet':
        if idx_images is not None:
            n_samples_org = copy.deepcopy(n_samples)
            n_samples = 1000
        images = load_imagenet_data(root=dataset.root_data, n=n_samples, train=train, format_numpy=format_numpy,
                                    image_size=dataset.image_size)
    else:
        raise ValueError(f'dataset = {dataset.name} not available.')

    if idx_images is not None:
        images = [images[i] for i in idx_images][:n_samples_org]
        assert len(images) == n_samples_org, \
            f'Could not load enough images with the appropriate n_superpixel.\n ' \
            f'n = {len(images)} are available.'

    return images


def load_model(model_cfg: ModelConfig, dataset_name: str) -> Model:
    if model_cfg.name == 'FullyConnected' and dataset_name == 'mnist':
        from src.models.mnist_lightning import load_mnist_model
        path_to_model = Path(get_original_cwd()) / "../src/models/model_mnist.ckpt"
        model_cfg = load_mnist_model(path_to_model=str(path_to_model))
    elif dataset_name == 'imagenet':
        model_cfg = load_imagenet_model(cfg_model=model_cfg)
    else:
        raise ValueError(f'Model does not exists: {model_cfg.name} for dataset {dataset_name}.')
    return model_cfg


def load_imputer(imputer_cfg: ImputerConfig, dataset: DatasetConfig) -> Imputer:
    """Returns 'impute_fn' a callable which can impute images from the current dataset."""
    if imputer_cfg.name == 'TrainSet':
        train_images_np = load_data(dataset=dataset, n_samples=imputer_cfg.n_train, train=True, format_numpy=True)
        imputer = TrainSetImputer(train_data=train_images_np)
    elif imputer_cfg.name == 'ColorHistogram' and dataset.name == 'imagenet':
        train_images_np = load_data(dataset=dataset, n_samples=100, train=True, format_numpy=True)
        imputer = ColorHistogramImputer(train_data=train_images_np)
    elif imputer_cfg.name == 'ConstantValueImputer':
        imputer = ConstantValueImputer(constant=imputer_cfg.value)
    elif imputer_cfg.name == 'cv2' and dataset.name == 'imagenet':
        from conditional_explainer.imputers.opencv_inpainting import OpenCVInpainting
        imputer = OpenCVInpainting(normalize_image=normalize_image, unnormalize_image=unnormalize_image,
                                   inpainting_algorithm=imputer_cfg.inpainting_algorithm)
    elif imputer_cfg.name.split("_")[0] == 'diffusion' and dataset.name == 'imagenet':
        # input_shape = (batch_size, n_channel, height, width)
        # config_file = '<path_to_config_files>/confs/test_inet256_thin.yml'
        # https://github.com/andreas128/RePaint/tree/main/confs
        from conditional_explainer.imputers.diffusion_based_imputer import DiffusionImputer
        # imputer_kwargs = {'diffusion_steps': 1000, 'time_steps': 250, 'n_resampling': 1, 'class_conditioning': False}
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config_file = (Path(get_original_cwd()).parent / imputer_cfg.relative_config_path).absolute()
        root_repaint_file = (Path(get_original_cwd()).parent / imputer_cfg.relative_path_to_repaint_root).absolute()
        imputer = DiffusionImputer(batch_size=imputer_cfg.batch_size, shape_input=(imputer_cfg.batch_size, 256, 256, 3),
                                   image_size=dataset.image_size,
                                   device=device, config_file=config_file,
                                   root_repaint=root_repaint_file,
                                   diffusion_steps=imputer_cfg.diffusion_steps, time_steps=imputer_cfg.time_steps,
                                   n_resampling=imputer_cfg.n_resampling,
                                   class_conditioning=imputer_cfg.class_conditioning,
                                   method_matching_internal_shape=imputer_cfg.method_matching_internal_shape
                                   )
        print('finished loading diffusion imputer.')
    elif imputer_cfg.name == 'internal':        # this is just a placeholder
        imputer = IdentityImputer()
        imputer.imputer_name = 'internal'
    else:
        raise NotImplementedError(f'Imputer available: {imputer_cfg.name}')
    return imputer


def check_redundant_keys(cfg_explainer: ConfigExplainer) -> bool:
    # check whether unwantend keys exist in cfg_explainer.
    if cfg_explainer.keys() & {'segmentation', 'imputer', 'cardinality_coalitions'} != set():
        return False
    else:
        return True


def load_explainer(cfg_explainer: ConfigExplainer, dataset: DatasetConfig, model: Model) -> ImageExplainer:
    if cfg_explainer.name not in ['PredDiff', 'Shapley values', 'KernelSHAP']:
        assert check_redundant_keys(cfg_explainer), f'Remove redundant keys from explainer cfg. \n{cfg_explainer}'

    if cfg_explainer.name in ['PredDiff', 'Shapley values', 'KernelSHAP']:
        imputer = load_imputer(imputer_cfg=cfg_explainer.imputer, dataset=dataset)

        # load images to initialize conditional_explainer
        train_images_np = load_data(dataset=dataset, n_samples=10, train=True, format_numpy=True)
        if cfg_explainer.name == 'PredDiff':
            explainer = PredDiff(model=model, conf_explainer=cfg_explainer, images_np=train_images_np, imputer=imputer)
        elif cfg_explainer.name == 'Shapley values':
            explainer = ShapleyValues(model=model, conf_explainer=cfg_explainer, images_np=train_images_np, imputer=imputer)
        elif cfg_explainer.name == 'KernelSHAP':
            explainer = KernelSHAP(model=model, conf_explainer=cfg_explainer, images_np=train_images_np, imputer=imputer)
        else:
            assert False, 'Check leading if statement for consistency.'

        if imputer.imputer_name == 'internal':
            assert model.model_name == 'Madry_ViT'

            def occluded_model(data: NDArray, segmentation_coalitions: NDArray) -> NDArray:
                data_broadcasted = np.broadcast_to(data, shape=segmentation_coalitions.shape)
                flat_shape = (-1, *data.shape[-3:])
                occluded_predictions_flat = [model.occluded_model_fn(sample, segmentation) for sample, segmentation in
                                             zip(data_broadcasted.reshape(flat_shape), segmentation_coalitions.reshape(flat_shape))]
                occluded_predictions = np.stack(occluded_predictions_flat).reshape((*segmentation_coalitions.shape[:-3], -1))
                return occluded_predictions

            explainer.core.external_occluded_model_fn = occluded_model

        if dataset.name == 'imagenet':
            explainer.batch_size = 16

    elif cfg_explainer.name == 'IntegratedGradients':
        from src.explainers.captum import IntegratedGradients
        explainer = IntegratedGradients(model=model, conf_explainer=cfg_explainer)

    elif cfg_explainer.name == 'Gradients':
        from src.explainers.captum import Gradients
        explainer = Gradients(model=model, conf_explainer=cfg_explainer)

    elif cfg_explainer.name == 'InputXGradients':
        from src.explainers.captum import InputXGradients
        explainer = InputXGradients(model=model, conf_explainer=cfg_explainer)

    elif cfg_explainer.name == 'CaptumShapleyValues':
        from src.explainers.captum import CaptumShapleyValues
        explainer = CaptumShapleyValues(model=model, conf_explainer=cfg_explainer)

    elif cfg_explainer.name == 'zennit':
        from src.explainers.zennit import ZennitExplainer
        explainer = ZennitExplainer(model=model, conf_explainer=cfg_explainer)

    elif cfg_explainer.name == 'Gradient':
        from src.explainers.zennit import ZennitGradient
        explainer = ZennitGradient(model=model, conf_explainer=cfg_explainer)

    else:
        raise ValueError(f'Explainer not implemented: {cfg_explainer.name}')
    return explainer
