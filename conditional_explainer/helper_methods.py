"""Contains helper function which are shared between different attributions methods."""
import numpy as np
from scipy.special import binom

from typing import List, Set
from numpy.typing import NDArray


def calc_n_interactions(n_feature: int, interaction_depth: int) -> int:
    """
    Calculate number of n-point interactions up to the maximum interaction_depth.
    Easily verified with Pascal's triangle.
    """
    list_n_interaction = [binom(n_feature, i) for i in range(interaction_depth + 1)]
    return int(np.sum(list_n_interaction))


def generate_feature_interaction_algebra(target_features: Set[int], interaction_depth: int = -1) -> List:
    """Generates combinatorics of all target features, i.e. all n-point interactions.

    Args:
        target_features: array of positive integer, [1, 2, 3]
        interaction_depth: maximum number of interacting features.
            If not specified set to -> interaction_depth=len(target_features)

    Returns:
        list_occluded_features as a list: ['empty_set', '1', '2', '3', '1^2', '1^3', '2^3', '1^2^3']
    """
    list_occluded_features = [f'empty_set']

    interaction_depth = len(target_features) if interaction_depth == -1 else interaction_depth

    def add_features(current_feature_interactions: List, new_feature: int) -> List:
        assert new_feature >= 0
        new_feature_interactions = []
        for current_feature in current_feature_interactions:
            if current_feature == 'empty_set':
                new_feature_interactions.append(f'{new_feature}')
            else:
                new_feature_interactions.append(f'{current_feature}^{new_feature}')
        return new_feature_interactions

    for new_target_feature in target_features:
        additional_feature_combinations = add_features(list_occluded_features, new_target_feature)
        list_occluded_features = list_occluded_features + additional_feature_combinations
        list_occluded_features.sort(key=lambda key: len(key.split('^')))   # sort according to # interacting features
        list_occluded_features = [key for key in list_occluded_features
                                            if len(key.split('^')) <= interaction_depth]

    return list_occluded_features


def generate_footprints(list_occluded_features: List) -> NDArray:
    """Generates a footprint for each feature interaction."""
    list_footprint = []
    assert list_occluded_features[0] == 'empty_set'
    n_int = len(list_occluded_features)

    for current_feature_interaction in list_occluded_features[1:]:
        footprint_np = np.zeros(n_int)                                      # as default no features contribute

        # generate all combinations of currently interacting features
        # 1^2^3 -> [1, 2, 3, 1^2, 1^3, 2^3, 1^2^3]
        list_split_features = current_feature_interaction.split('^')        # generate a list of all feature involved
        np_split_features = set([int(element) for element in list_split_features])     # convert feature keys to np
        all_target_sets = generate_feature_interaction_algebra(target_features=np_split_features)

        global_sign = - (-1) ** len(list_split_features)        # alternating sign depending on #features involved
        for interacting_feature in all_target_sets:             # include all_target_sets in footprint
            # alternating sign depending on number of interacting_features

            if interacting_feature == 'empty_set':      # 0 -> +
                sign = 1
            else:                                       # 1 -> -, 2 -> -, 1^2 -> +, 1^2^3 -> -, 1^2^3^4 -> +
                sign = (-1) ** (len(interacting_feature.split('^')))
            mask = np.array(list_occluded_features) == interacting_feature
            footprint_np[mask] = sign * global_sign

        list_footprint.append(footprint_np)

    footprints = np.stack(list_footprint)
    return footprints


def incorporate_interacting_features(segmentation_raw_coalitions: NDArray, list_occluded_features: List[str]) \
        -> NDArray:
    """
    Add appropriate interacting segments to all coalitions.

    Args:
        segmentation_raw_coalitions: shape: (n_coalitions, *shape_input, *shape_x)
        list_occluded_features: len(list_occluded_features) = n_int

    Returns: shape: (n_int, n_coalitions, *shape_input, *shape_x)
    """
    n_interaction = len(list_occluded_features)

    # allocate interacting segmentations
    segmentation_interaction_coalitions = np.stack([segmentation_raw_coalitions for _ in range(n_interaction)])

    # indicate feature_interaction with negative feature label within the complete segment
    for i, feature_interaction in enumerate(list_occluded_features):
        if feature_interaction == 'empty_set':  # keep full segmentation
            continue
        list_individual_features = feature_interaction.split(sep='^')
        for feature in list_individual_features:  # replace each segments with '-1'
            mask = int(feature) == segmentation_raw_coalitions
            segmentation_interaction_coalitions[i, mask] *= -1  # flip feature label
    return segmentation_interaction_coalitions
