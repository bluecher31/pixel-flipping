import numpy
import numpy as np
from scipy.special import binom

from typing import List, Set
from numpy.typing import NDArray


def shap_kernel_weight(n_features: int, size_coalition: int) -> int:
    """
    Computes (N-1)/[binom(N, |S|) * |S| * (N - |S|)] depending on size_coalition (|S|) and n_features (N) which is
    appropriate for the weighted linear regression.
    """
    if size_coalition == n_features or size_coalition == 0:
        return 1_000  # large constant to proxy infinity
    else:
        div = binom(n_features, size_coalition) * size_coalition * (n_features - size_coalition)
        res = (n_features - 1) / div
        return res


def _preprocess_cardinality_coalitions(cardinality_coalitions: List[int], n_features: int) -> [List[int], str]:
    """
    Perform consistency check and invert cardinalities if necessary. 
    
    Args:
        cardinality_coalitions: list of positive/negative or mixed integers
        n_features: total number of features involved
        
    Returns:
        [cardinality_coalitions with only positive integers, description of changes performed]
    """
    if cardinality_coalitions is None:  # use all coalitions
        cardinality_coalitions = [i for i in range(1, n_features)]
        type_cardinality = 'positive'
    else:                               # restrict to the requested cardinalities of coalitions
        np_cardinality = np.array(cardinality_coalitions)
        if np.alltrue(np_cardinality > 0):
            type_cardinality = 'positive'
        elif np.alltrue(np_cardinality < 0):
            type_cardinality = 'negative'
            cardinality_coalitions = [-cardinality for cardinality in cardinality_coalitions]
        else:
            type_cardinality = 'mixed'
            cardinality_positive = [c for c in cardinality_coalitions if c > 0]
            cardinality_negative = [n_features - abs(c) for c in cardinality_coalitions if c <= 0]
            cardinality_coalitions = [*cardinality_positive,
                                      *[c for c in cardinality_negative if c >= 0]]

    # remove redundant cardinalities (too large and double entries)
    cardinality_coalitions = list(set([c for c in cardinality_coalitions if c <= n_features]))

    return cardinality_coalitions, type_cardinality


def _postprocess_coalitions(list_coalitions: List[Set[int]], features: Set[int], type_cardinality: str) \
        -> List[Set[int]]:
    """Sort according to cardinality. Invert coalitions (complement wrt features) if necessary."""
    if type_cardinality == 'negative':  # calculate complement w.r.t. all features
        list_coalitions = [features.difference(S) for S in list_coalitions]
    else:
        pass

    list_coalitions.sort(key=lambda s: len(s))
    return list_coalitions


def power_set_coalitions(features: Set[int], cardinality_coalitions: List[int] = None) \
        -> List[Set[int]]:
    """
    Creates the power set for the provided features.


    Args:
        features: positive integers, shape: (n_features)
        cardinality_coalitions: cardinality of coalitions used to calculate the Shapley values (N: total # features).
            None: all coalitions -> [1, 2, 3,..., N-1]
            [1, 2]: |S| = 1 or |S| = 2
            [-1, -3]: |S| = N - 1 or |S| = N -3
    Returns:
        [{1}, {2}, {3}, {1, 2}, {1, 3}, {2, 3}] for features = [1, 2, 3] and  cardinality_coalitions = [1, 2]
        In any case this excludes the empty_set {} and full set {1, 2,...,N}
    """
    cardinality_coalitions, type_cardinality = _preprocess_cardinality_coalitions(cardinality_coalitions,
                                                                                     n_features=(len(features)))
    maximal_cardinality = max(np.abs(cardinality_coalitions))

    n_possible_coalitions = binom(len(features), maximal_cardinality)
    assert n_possible_coalitions < 1E5, \
        f'Algorithm calculates too many coalitions. n_possible_coalitions={n_possible_coalitions}'

    list_power_set = [set()]
    for f in features:
        list_new_coalitions = []
        for coalition in list_power_set:
            temp_set = coalition.copy()
            temp_set.add(int(f))
            if len(temp_set) > maximal_cardinality:
                continue
            list_new_coalitions.append(temp_set)
        list_power_set.extend(list_new_coalitions)

    list_coalitions = [s for s in list_power_set if len(s) in cardinality_coalitions]

    list_coalitions = _postprocess_coalitions(list_coalitions, features, type_cardinality)
    return list_coalitions


def random_coalitions(features: Set[int], n_coalitions: int, cardinality_coalitions: List[int] = None,
                      prior_cardinality: str = 'binom', symmetric_coalitions: bool = False) \
        -> List[Set[int]]:
    """
    Creates a list of random coalitions with probabilistic cardinalities and features.
    Excludes empty and full set.

    Args:
        prior_cardinality: prior distribution over coalition cardinalities.
            'binom' necessary to ensure a consistent estimator
        symmetric_coalitions: paired sampling as proposed in Covert21 (Improved KernelSHAP)
            Include all complement coalitions S & \bar{S}.
    """
    cardinality_coalitions, type_cardinality = _preprocess_cardinality_coalitions(cardinality_coalitions,
                                                                                  n_features=(len(features)))

    # if (set() in cardinality_coalitions) or (0 in cardinality_coalitions) or len(features) in cardinality_coalitions:
    #     raise ValueError('Empty and full coalition are not allowed.')
    if len(cardinality_coalitions) > 0:
        if prior_cardinality == 'uniform':
            random_cardinalities = np.random.choice(cardinality_coalitions, n_coalitions, replace=True)
        elif prior_cardinality == 'binom':  # sample depending on frequency in binom coefficient
            frequency_cardinality = [binom(len(features), cardinality) for cardinality in cardinality_coalitions]
            random_cardinalities = np.random.choice(cardinality_coalitions, n_coalitions,
                                                    p=frequency_cardinality / np.sum(frequency_cardinality), replace=True)
        else:
            raise NotImplementedError(f'prior_cardinality = {prior_cardinality} is not implemented.')
    else:
        random_cardinalities = np.zeros(n_coalitions, dtype=int)

    list_coalitions = [set(np.random.choice(list(features), size=cardinality, replace=False))
                       for cardinality in random_cardinalities]
    # for i in range(n_coalitions):
    #     coalition = set(np.random.choice(list(features), size=cardinality, replace=False))
    #     list_coalitions.append(coalition)

    if symmetric_coalitions is True:
        list_coalitions_base = list_coalitions[:int(len(list_coalitions) / 2)]
        list_coalitions_paired = [features.difference(S) for S in list_coalitions_base]
        list_coalitions = [*list_coalitions_base, *list_coalitions_paired]
        type_cardinality = 'mixed'

    list_coalitions = _postprocess_coalitions(list_coalitions, features, type_cardinality)
    return list_coalitions


def convert_segmentation_to_list(segmentation_coalitions: NDArray) -> List[Set]:
    """
    Convert segmentation_coalition: NDArray into list_coalitions: List[Set]

    Args: coalition members are positive-valued, non-member negative-valued,
        shape: (n_coalitions, *shape_input, *shape_x), dtype=int

    Returns:
        list_coalitions: len(list_coalitions): n_coalitions
    """
    list_coalitions = []
    for coalition in segmentation_coalitions[0]:
        assert np.alltrue(coalition[0] == coalition), 'coalition needs to be consistent along the input dimension.'
        mask_positive = coalition > 0
        list_coalitions.append(set(coalition[mask_positive]))
    return list_coalitions


def convert_features_into_mask(features: Set[int], segmentation: NDArray[int]) -> NDArray[bool]:
    mask_stacked = np.stack([int(feature) == segmentation for feature in features])
    mask = np.sometrue(mask_stacked, axis=0)
    return mask


def convert_coalitions_to_segmentation(segmentation: NDArray, list_coalitions: List[Set[int]]) \
        -> NDArray:
    """
    Args:
        segmentation: shape: (*shape_input, *shape_x)
        list_coalitions: List[Set], i.e. [empty_set, set(features), S3,..., S_{n_coalitions}]
    Returns:
        segmentation_coalitions: negative values refer to be occluded features
            shape: (1, n_coalitions, *shape_input, *shape_x), dtype=int,
    """
    # TODO: check convention, seems to be inconsistent, {} -> should correspond to original prediction

    def get_segmentation_coalition(coalition: Set[int]) -> NDArray:
        """Returns the segmentation corresponding to the coalition."""
        segmentation_coalition = -segmentation.copy()
        if coalition == set():      # empty set, do not include any feature
            pass
        else:
            mask = convert_features_into_mask(features=coalition, segmentation=segmentation)
            segmentation_coalition[mask] *= -1       # include feature segment
        return segmentation_coalition

    segmentation_coalitions = np.stack([get_segmentation_coalition(coalition) for coalition in list_coalitions])

    return segmentation_coalitions[None]
