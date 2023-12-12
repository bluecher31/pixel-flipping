"""Contains simple helper functions which are used by the base_explainer class:
        (i) format check of input variables/callables
        (ii) laplace smoothing
"""

import numpy as np
from numpy.typing import NDArray

from typing import Callable, Tuple

# cannot import from abstract_base_explainer due to circular import, fix would be a dedicated file with definitions
ImputeFn = Callable[[NDArray, NDArray[int], int], NDArray]
# descriptive signature: [[data, segmentation_coalition, n_imputations], imputations]
ModelFn = Callable[[NDArray], NDArray]


def check_impute_fn(impute_fn: ImputeFn, data: NDArray, segmentation_coalitions: NDArray, n_imputations: int):
    """Checks whether (i) return shape of impute_fn is as expected and (ii) only correct features are altered."""
    x_hat = impute_fn(data, segmentation_coalitions, n_imputations)  # (n_int, n_imp, n_s, *data.shape)
    n_mask = segmentation_coalitions.shape[0]
    assert x_hat.shape == (n_imputations, n_mask, *data.shape), 'Incorrect shape of x_hat.'
    assert data.shape == segmentation_coalitions.shape[1:], 'Sanity check.'
    assert np.issubdtype(segmentation_coalitions.dtype, int), 'Please segmentation needs to be of type int.'

    mask_impute = segmentation_coalitions <= -1
    mask_original_feature = np.broadcast_to(np.invert(mask_impute[None]), x_hat.shape)
    x_broadcast = np.broadcast_to(data, x_hat.shape)
    mask_test = x_hat == x_broadcast
    # check only for feature which are not to be imputed
    assert np.alltrue(mask_test[mask_original_feature] == mask_original_feature[mask_original_feature]), \
        'Feature changes detected which belong to fixed unaltered coalition.'


def partial_check_impute_fn(impute_fn: ImputeFn, data: NDArray):
    """Generates a random segmentation mask to call 'check_impute_fn'. """
    n_int = 2
    n_coalitions = 10
    segmentation_coalitions = np.ones(data.size * n_int * n_coalitions, dtype=int)
    n_features_impute = 15
    segmentation_coalitions[:n_features_impute] = -(np.arange(n_features_impute) + 3)
    np.random.shuffle(segmentation_coalitions)
    segmentation_coalitions = segmentation_coalitions.reshape((n_int*n_coalitions, *data.shape))
    assert segmentation_coalitions.sum() != 0
    check_impute_fn(impute_fn=impute_fn, data=data,
                    segmentation_coalitions=segmentation_coalitions, n_imputations=5)


def get_and_check_shapes(model_fn: ModelFn, data: NDArray) -> (Tuple, int):
    """Checks model_fn compatibility of and returns shape of input sample and output."""
    # test model_fn function
    # shape_input = samples.shape
    predictions = model_fn(data)
    shape_input = predictions.shape[:-1]       # shape of arbitrary allocation of data sample
    assert len(shape_input) < len(data.shape), 'model_fn adds dimensions to the data. '
    n_classes = predictions.shape[-1]
    assert n_classes > 0
    shape_x = data.shape[len(shape_input):]         # shape of single data sample
    assert predictions.shape == (*shape_input, n_classes)

    # model_fn preserves shape
    # test_sample = data[None, None]
    # test_sample_prediction = model_fn(test_sample)
    # assert test_sample_prediction.shape == (1, 1, *shape_input, n_classes), 'model_fn changes shape of input.'

    return shape_x, n_classes


def laplace_smoothing(probability: NDArray, n_train: int, n_classes: int, alpha=1) -> NDArray:
    """
    Performs smoothing to remove zero probabilities.
    Args:
        probability: array with probabilities
        n_train: number of training samples
        n_classes:
        alpha:

    Returns:
        smoothed probabilities
    """
    smoothed_probabilities = (probability * n_train + alpha) / (n_train + n_classes)
    return smoothed_probabilities


def organize_key(key: str) -> str:
    """Sort interacting attribution keys ('2^4^1^3' -> '1^2^3^4'). All others are just passed."""

    key_splitted = key.split('^')
    test_for_digits = [tmp.isdigit() for tmp in key_splitted]
    if np.alltrue(test_for_digits):  # detected attribution key
        # sort digits
        key_splitted.sort(key=lambda str_key: int(str_key))
        key_sorted = key_splitted[0]
        if len(key_splitted) > 1:  # two or more interacting features
            for key_tmp in key_splitted[1:]:
                key_sorted += f'^{key_tmp}'
    else:
        key_sorted = key
    return key_sorted
