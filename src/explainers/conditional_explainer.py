import numpy as np

import torch

from src.explainers.helper_images import load_segmentor
from src.interface import ImageExplainer, Image, Model, ConfigExplainer

from typing import List, Dict, Union
from numpy.typing import NDArray
from abc import ABC, abstractmethod
from dataclasses import dataclass

from conditional_explainer import preddiff, shapley, kernelshap
from conditional_explainer.base_explainer import BaseExplainer
from conditional_explainer.imputers.abstract_imputer import Imputer
from src.config.config import ImputerConfig, SegmentationConfig


@dataclass
class ConfigConditionalExplainer(ConfigExplainer):
    n_eval: int
    imputer: ImputerConfig
    segmentation: SegmentationConfig


@dataclass
class PropertiesShapleyValues(ConfigConditionalExplainer):
    cardinality_coalitions: Union[List[int], None]
    symmetric_coalitions: bool      # enforce complementary sampled coalitions


@dataclass
class PropertiesKernelSHAP(PropertiesShapleyValues):
    ridge_parameter: float


def convert_dict_to_heatmap(dict_attributions: Dict, image: Image) -> NDArray:
    """Dictionary containing attributions for all superpixels and the corresponding segmentation."""
    segmentation = np.squeeze(dict_attributions['segmentation'])
    heatmap = np.zeros_like(segmentation, dtype=float)

    for seg in np.unique(segmentation):
        all_relevances = dict_attributions[f'{seg}']
        target_relevance = float(np.squeeze(all_relevances)[image.label])
        mask = segmentation == seg
        heatmap[mask] = target_relevance
    return heatmap


class AbstractConditionalExplainer(ImageExplainer, ABC):
    def __init__(self, model: Model, conf_explainer: ConfigConditionalExplainer, images_np: NDArray, imputer: Imputer):
        assert conf_explainer.imputer.name == imputer.imputer_name, f'Provided properties and imputer do not match.\n' \
                                                                f'{conf_explainer.imputer.name} == {imputer.imputer_name}'
        super().__init__(model=model, conf_explainer=conf_explainer)
        self.generate_segmentation = load_segmentor(cfg_segmentation=conf_explainer.segmentation)

        self.explainer_name, self.core = self.get_explainer(images_np=images_np, imputer=imputer)
        self.core.batch_size = 32

    @abstractmethod
    def get_explainer(self, images_np: NDArray, imputer: Imputer) -> [str, BaseExplainer]:
        """Wrapper to load the core explainer functionality. str: explainer name"""

    def get_heatmap(self, image: Image) -> NDArray:
        """Calculate a heatmap (relevance for each pixel) based on the base explainer."""
        segmentation = self.generate_segmentation(image.image)

        self.core.interaction_depth = 1
        attributions_for_all_superpixel = self.core.attribution(data=image.image[np.newaxis],
                                                                segmentation=segmentation[np.newaxis],
                                                                target_features=set(segmentation.flatten()))

        heatmap = convert_dict_to_heatmap(attributions_for_all_superpixel, image)
        return heatmap


class PredDiff(AbstractConditionalExplainer):
    def get_explainer(self, images_np: NDArray, imputer: Imputer) -> [str, BaseExplainer]:
        core = preddiff.PredDiff(model_fn=self.model.predict, imputer=imputer,
                                 n_eval=self.conf.n_eval, data=images_np)
        return 'PredDiff', core


class ShapleyValues(AbstractConditionalExplainer):
    def __init__(self, model: Model, conf_explainer: PropertiesShapleyValues, images_np: NDArray, imputer: Imputer):
        """
        cardinality_coalitions: |S| which are used to calculate the Shapley values (N: total number of features).
                None: all coalitions -> [1, 2, 3,..., N]
                [1, 2]: |S| = 1 or |S| = 2
                [-1, -3]: |S| = N - 1 or |S| = N -3
        """
        super().__init__(model=model, conf_explainer=conf_explainer, images_np=images_np, imputer=imputer)

    def get_explainer(self, images_np: NDArray, imputer: Imputer) -> [str, BaseExplainer]:
        core = shapley.ShapleyValues(model_fn=self.model.predict, imputer=imputer,
                                     n_eval=self.conf.n_eval, data=images_np,
                                     cardinality_coalitions=self.conf.cardinality_coalitions,
                                     symmetric_coalitions=self.conf.symmetric_coalitions)
        return 'Shapley values', core

    def get_heatmap(self, image: Image) -> NDArray:
        """Calculate a heatmap (relevance for each pixel) based on the base explainer."""
        segmentation = self.generate_segmentation(image.image)
        with torch.no_grad():
            attributions_for_all_superpixel = {}
            for s in set(segmentation.flatten()):       # iterate all superpixels
                attributions_s = self.core.attribution(data=image.image[np.newaxis],
                                                       segmentation=segmentation[np.newaxis],
                                                       target_features={s})
                attributions_for_all_superpixel.update(attributions_s)          # collect attributions

        heatmap = convert_dict_to_heatmap(attributions_for_all_superpixel, image)
        return heatmap


class KernelSHAP(AbstractConditionalExplainer):
    def __init__(self, model: Model, conf_explainer: PropertiesShapleyValues, images_np: NDArray, imputer: Imputer):
        """
        cardinality_coalitions: |S| which are used to calculate the Shapley values (N: total number of features).
                None: all coalitions -> [1, 2, 3,..., N]
                [1, 2]: |S| = 1 or |S| = 2
                [-1, -3]: |S| = N - 1 or |S| = N -3
        """
        super().__init__(model=model, conf_explainer=conf_explainer, images_np=images_np, imputer=imputer)

    def get_explainer(self, images_np: NDArray, imputer: Imputer) -> [str, BaseExplainer]:
        core = kernelshap.KernelSHAP(model_fn=self.model.predict, imputer=imputer,
                                     n_eval=self.conf.n_eval * self.conf.n_superpixel, data=images_np,
                                     cardinality_coalitions=self.conf.cardinality_coalitions,
                                     symmetric_coalitions=self.conf.symmetric_coalitions,
                                     ridge_parameter=self.conf.ridge_parameter)
        return 'KernelSHAP', core

