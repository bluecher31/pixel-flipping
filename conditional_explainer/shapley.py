import numpy as np
from numpy.typing import NDArray

from typing import Callable, List, Set
from conditional_explainer.imputers.abstract_imputer import Imputer

from conditional_explainer import helper_methods, helper_kernelshap
from conditional_explainer.derived_explainer_targeted import TargetedAttributionsMethod


def _get_unique_features(list_features: List[str]) -> NDArray:
    """Extracts all unique features contained in list_features. """
    all_features = np.concatenate([features.split(sep='^') for features in list_features])
    unique_features = np.unique(np.array(all_features, dtype=np.int))
    return unique_features


def add_non_interacting_features(list_features: List[str], new_feature_set: NDArray) -> List[str]:
    """Uniformly sample a sub_feature_set  from new_features_set and add to each element in list_features."""
    set_features = np.append(new_feature_set, [-1])
    np.random.shuffle(set_features)
    subset_feature = set_features[:set_features.argmin()]  # argmin corresponds to -1

    if len(subset_feature) > 0:     # only if features need to be added
        # convert to feature identifier
        new_features = f'{subset_feature[0]}'
        for feature in subset_feature[1:]:
            new_features += f'^{feature}'

        assert list_features[0] == 'empty_set'

        list_features_new = [new_features, *[old_features+f'^{new_features}' for old_features in list_features[1:]]]
    else:
        list_features_new = list_features
    return list_features_new


def generate_footprints_shapley(list_occluded_features: List[str], n_coalitions: int) -> NDArray:
    """Generates a footprint for each feature interaction."""
    list_footprint = []
    assert list_occluded_features[0] == 'empty_set'
    n_int = len(list_occluded_features)
    all_features_np = _get_unique_features(list_features=list_occluded_features[1:])

    for current_feature_interaction in list_occluded_features[1:]:    # calculate attribution for these features
        footprint_np = np.zeros((n_int, n_coalitions))      # as default no features contribute

        # generate all combinations of currently interacting features
        # 1^2^3 -> [1, 2, 3, 1^2, 1^3, 2^3, 1^2^3]
        # all occluded subset contribute to attributions -> list_contributing_feature_sets
        np_split_features = _get_unique_features([current_feature_interaction])   # convert feature keys to np
        list_contributing_feature_sets = helper_methods.generate_feature_interaction_algebra(
            target_features=set(np_split_features))

        # TODO: sample interacting features appropriately within the corresponding footprint
        #  undo this to calculate shielded shapley values
        # Virtually add non-interacting features to coalition by changing the footprint
        current_features_np = _get_unique_features(list_features=[current_feature_interaction])
        non_interacting_features_np = np.setdiff1d(all_features_np, current_features_np, assume_unique=True)

        # alternating sign depending on number of base_features
        global_sign = - (-1) ** len(np_split_features)  # alternating sign depending on #features involved
        for i in range(n_coalitions):       # randomly add non-interacting features to coalition
            list_features_footprint = add_non_interacting_features(list_contributing_feature_sets,
                                                                   non_interacting_features_np)

            for base_features, coalition_features in zip(list_contributing_feature_sets, list_features_footprint):
                if base_features == 'empty_set':      # 0 -> +
                    sign = 1
                else:                                       # 1 -> -, 2 -> -, 1^2 -> +, 1^2^3 -> -, 1^2^3^4 -> +
                    sign = (-1) ** (len(base_features.split('^')))
                mask_list = [set(features) == set(coalition_features) for features in list_occluded_features]
                footprint_np[mask_list, i] = sign * global_sign

        assert footprint_np.shape == (len(list_occluded_features), n_coalitions)
        list_footprint.append(footprint_np)
    return np.stack(list_footprint)


def get_coalitions(spectator_features: NDArray, cardinality_coalitions: List[int],
                   n_coalitions: int, symmetric_coalitions: bool) \
        -> List[Set[int]]:
    """Sample the a set of coalitions."""
    if cardinality_coalitions is None:
        all_features = np.append(spectator_features, [-1])
        list_coalitions = []
        for _ in range(n_coalitions):
            np.random.shuffle(all_features)
            coalition = all_features[:all_features.argmin()]  # argmin corresponds to -1
            list_coalitions.append(set(coalition))
    else:
        # raise negative cardinalities since all target features are already removed
        cardinality_coalitions = [c if c > 0 else c + 1 for c in cardinality_coalitions]

        list_coalitions = helper_kernelshap.random_coalitions(
            features=set(spectator_features), cardinality_coalitions=cardinality_coalitions, n_coalitions=n_coalitions,
            prior_cardinality='binom', symmetric_coalitions=symmetric_coalitions)

    return list_coalitions


class ShapleyValues(TargetedAttributionsMethod):
    """"Implements on-manifold attributions based on arXiv:2102.13519"""
    def __init__(self, model_fn: Callable, imputer: Imputer, n_eval: int, data: NDArray,
                 cardinality_coalitions: List[int] = None, symmetric_coalitions: bool = False):
        """

        Args:
            model_fn: callable model, shapes: (*shape_input, *shape_x) -> (*shape_input, n_classes),
            imputer: models conditional distribution which is used to masks features and build an occluded_model_fn.
                function signature: [data: NDArray, segmentation_coalition: NDArray, n_imputations: int] -> imputations,
                data.shape: (*shape_input, *shape_x)
                segmentation_coalition.shape = (n_masks, *shape_input, *shape_x)
                imputations.shape = (n_imputations, n_masks, *shape_input, *shape_x)
            n_eval: approximate number of model calls
            data: arbitrary test samples, required to probe model_fn/impute_fn for clf vs. reg and shapes,
                shape: (*shape_input, *shape_x)
            cardinality_coalitions: |S| which are used to calculate the Shapley values (N: total number of features).
                None: all coalitions -> [1, 2, 3,..., N]
                [1, 2]: |S| = 1 or |S| = 2
                [-1, -3]: |S| = N - 1 or |S| = N -3
                In particular this means that |S| = -1 is equivalent to PredDiff.
            symmetric_coalitions: always complementary coalitions to improve convergence as
                proposed in Covert&Lee: Improved KernelSHAP @ AISTATS 21
        """
        super().__init__(model_fn=model_fn, imputer=imputer, n_eval=n_eval, data=data)
        self.n_imputations = 1
        self.explainer_name = 'ShapleyValues'
        self.cardinality_coalitions = cardinality_coalitions
        self.symmetric_coalitions = symmetric_coalitions

    def _footprint_fn(self, target_features: Set[int]) -> [NDArray, List[str]]:
        """Generates combinatorics between all target_features and the corresponding footprint measures."""
        if self.interaction_depth != -1:
            raise NotImplementedError('Only full interaction depth implemented.')
        list_occluded_features = helper_methods.generate_feature_interaction_algebra(
            target_features=target_features, interaction_depth=self.interaction_depth)

        footprints = generate_footprints_shapley(list_occluded_features=list_occluded_features,
                                                 n_coalitions=self.n_coalitions)

        return list_occluded_features, footprints

    def _generate_coalitions(self, segmentation: NDArray, list_occluded_features: List[str]) -> NDArray:
        """
        Coalitions based on feature_interactions and a single unique/deterministic base_coalition.
        These coalitions are converted into a segmentation mask.

        Returns:
            Array with shape (n_int, n_coalitions, *segmentation.shape), dtype=int
            To be occluded features are indicated with negative integer label.
            axis n_int: -1 are distributed according to list_occluded_features
            axis n_coalitions: random subsampled coalitions S
        """
        assert list_occluded_features[0] == 'empty_set'
        interacting_features = _get_unique_features(list_features=list_occluded_features[1:])

        assert len(np.unique(segmentation)) >= len(interacting_features)
        # set of possible coalitions, i.e. the complement of ll interacting_features.
        # here we follow the standard convention to  denote the set of spectator feature with S
        spectator_features = np.setdiff1d(np.unique(segmentation), interacting_features, assume_unique=True)
        mask_spectator = helper_kernelshap.convert_features_into_mask(features=set(spectator_features),
                                                                      segmentation=segmentation)

        seg_base_coalitions = np.array(np.broadcast_to(segmentation, shape=(self.n_coalitions, *segmentation.shape)))
        seg_base_coalitions[:, mask_spectator] *= -1            # no feature is present, default coalition = {}

        list_coalitions = get_coalitions(spectator_features=spectator_features,
                                         cardinality_coalitions=self.cardinality_coalitions,
                                         n_coalitions=self.n_coalitions, symmetric_coalitions=self.symmetric_coalitions)

        for i, coalition in enumerate(list_coalitions):
            if len(coalition) > 0:
                mask_stacked = np.stack([int(feature) == segmentation for feature in coalition])
            else:
                mask_stacked = np.zeros_like(segmentation, dtype=bool)[None]
            mask = np.sometrue(mask_stacked, axis=0)
            seg_base_coalitions[i, mask] *= -1

        seg_interaction_coalitions = helper_methods.incorporate_interacting_features(
            segmentation_raw_coalitions=seg_base_coalitions, list_occluded_features=list_occluded_features)

        return seg_interaction_coalitions
