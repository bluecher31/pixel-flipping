from typing import Callable, List, Set
from numpy.typing import NDArray
from conditional_explainer.imputers.abstract_imputer import Imputer

from conditional_explainer import helper_methods
from conditional_explainer.derived_explainer_targeted import TargetedAttributionsMethod


class PredDiff(TargetedAttributionsMethod):
    """"Implements on-manifold attributions based on arXiv:2102.13519"""
    def __init__(self, model_fn: Callable, imputer: Imputer, n_eval: int, data: NDArray):
        super().__init__(model_fn=model_fn, imputer=imputer, n_eval=n_eval, data=data)
        self.n_imputations = self.n_eval
        self.explainer_name = 'PredDiff'

    def _footprint_fn(self, target_features: Set[int]) -> [List[str], NDArray]:
        """Generates combinatorics between all target_features and the corresponding footprint measures."""
        list_occluded_features = helper_methods.generate_feature_interaction_algebra(
            target_features=target_features, interaction_depth=self.interaction_depth)
        footprints = helper_methods.generate_footprints(list_occluded_features=list_occluded_features)
        footprints = footprints[..., None]        # add n_coalition=1 dimension
        return list_occluded_features, footprints

    def _generate_coalitions(self, segmentation: NDArray, list_occluded_features: List[str]) -> NDArray:
        """
        Coalitions based on feature_interactions and a single unique/deterministic base_coalition. See also base method.

        Args:
            segmentation: feature segmentation for data denoted by positive integers
                shape: (*shape_input, *shape_x), dtype=int
            list_occluded_features:

        Returns: segmentation_coalitions, shape: (n_int, n_coalitions, *segmentation.shape), dtype=int
        """
        assert self.n_coalitions == 1
        seg_base_coalitions = segmentation.reshape((self.n_coalitions, *segmentation.shape))

        seg_interaction_coalitions = helper_methods.incorporate_interacting_features(
            segmentation_raw_coalitions=seg_base_coalitions, list_occluded_features=list_occluded_features)

        return seg_interaction_coalitions
