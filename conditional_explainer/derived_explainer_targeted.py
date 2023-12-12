import numpy as np


from abc import abstractmethod

from conditional_explainer.imputers.abstract_imputer import Imputer
from conditional_explainer.base_explainer import BaseExplainer
from conditional_explainer.base_explainer import ModelFn, DictAttributions

from typing import List, Set
from numpy.typing import NDArray


class TargetedAttributionsMethod(BaseExplainer):
    def __init__(self, model_fn: ModelFn, imputer: Imputer, n_eval: int, data: NDArray):
        """
        Refer to base for details.

        Applies to methods which calculate attributions for each feature individually.
        """
        super().__init__(model_fn=model_fn, imputer=imputer, n_eval=n_eval, data=data)

    def _attribution(self, data: NDArray, segmentation: NDArray, target_features: Set[int]) -> DictAttributions:
        """
        See base class.
        """
        assert self.interaction_depth == -1 or 0 < self.interaction_depth <= len(target_features), \
            'Invalid interaction_depth.'

        # set-up all S combinatorics
        list_occluded_features, footprints = self._footprint_fn(target_features)
        self.n_int = len(list_occluded_features)  # abbr. for n_interaction
        assert footprints.shape == (self.n_int - 1, self.n_int, self.n_coalitions), 'Incorrect footprint shape.'

        segmentation_coalitions = self._generate_coalitions(segmentation, list_occluded_features)
        assert segmentation_coalitions.shape == (self.n_int, self.n_coalitions, *self.shape_input, *self.shape_x), \
            'Incorrect shape for segmentation_coalitions.'

        occluded_predictions = self._occluded_model_fn(data, segmentation_coalitions)
        # shape: (n_int, n_coalitions, *shape_input, n_classes)

        assert occluded_predictions.shape == \
               (segmentation_coalitions.shape[0], segmentation_coalitions.shape[1], *self.shape_input, self.n_classes),\
               'occluded_model_fn changed shape.'

        # aggregate occluded_predictions to obtain attributions
        dict_attributions = self._calculate_attributions(occluded_predictions=occluded_predictions,
                                                         footprints=footprints,
                                                         list_occluded_features=list_occluded_features)

        dict_information = {'target_features': target_features, 'data': data, 'segmentation': segmentation}
        dict_explanation = {**dict_attributions, **dict_information, **self.get_summary()}
        return dict_explanation

    def _calculate_attributions(self, occluded_predictions: NDArray, footprints: NDArray,
                                list_occluded_features: List[str]) -> DictAttributions:
        """
        Aggregates all individual occluded model_fn predictions into a set of attributions.

        Args:
            occluded_predictions: shape: (n_int, n_coalitions, *shape_input, *shape_x)
            footprints: len([f1, f2,...]) = n_int - 1, f1.
            list_occluded_features: len(list_occluded_features) = n_int,
        """
        dict_attributions = {}
        for footprint, key_attribution in zip(footprints, list_occluded_features[1:]):
            # n_int -> i, n_coalition -> j
            # compute attribution target function
            attributions_coalitions = np.einsum('ij,ij...->j...', footprint, occluded_predictions)
            attributions = attributions_coalitions.mean(axis=0)  # average all coalitions
            assert attributions.shape == (*self.shape_input, self.n_classes)
            dict_attributions[key_attribution] = attributions
        return dict_attributions

    @abstractmethod
    def _generate_coalitions(self, segmentation: NDArray, list_occluded_features: List[str]) -> NDArray:
        """
        Overwritten by individual explainers.

        Creates a segmentation mask where all occluded features are marked by a negative integer.
        Minimally all features labels included in list_occluded_features are flipped.
        Some methods such as Shapley values also flip additional features contained in (random) coalitions.

        Notes: occluded features are labeled by corresponding negative integer.
            The remaining segmentation is not changed.

        Args:
            segmentation: feature segmentation for data denoted by positive integers
                shape: (*shape_input, *shape_x), dtype=int
            list_occluded_features: list with keys. Each key contains features which need to be occluded.

        Returns: segmentation_coalitions, occluded features are indicated by a negative integer,
            coalitions may add additional features which are occluded,
            shape: (n_int, n_coalitions, *segmentation.shape), dtype=int
        """

    @abstractmethod
    def _footprint_fn(self, target_features: Set[int]) -> [List[str], NDArray]:
        """
        Overwritten by individual explainers.

        Creates a list_occluded_features and a corresponding numerical footprint how these occluded predictions need to
        be aggregated to calculate a single attribution.

        Args:
            target_features: shape: (n_targets)

        Returns:
            list_occluded_features: each element in list indicates which features are occluded from the model
                len(list_occluded_features) = n_int
            footprints: axis-meaning: (axis=0: n-point attributions, axis=1: how to sum up occluded predictions),
                shape: (n_int -1, n_int)
        """
