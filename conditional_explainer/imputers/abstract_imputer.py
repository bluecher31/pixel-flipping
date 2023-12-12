from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


def _check_input(data: NDArray, segmentation_coalitions: NDArray):
    # TODO: test for: data.shape = (*shape_input, *shape_x)
    if data.ndim + 1 == segmentation_coalitions.ndim and data.shape == segmentation_coalitions.shape[1:] is False:
        print('hello')
    assert data.ndim + 1 == segmentation_coalitions.ndim and data.shape == segmentation_coalitions.shape[1:], \
        f'Incompatible shapes. data.shape = (*shape_input, *shape_x) with at least one-dimensional shape_input. \n' \
        f'Provided: data.shape = {data.shape}, segmentation_coalitions.shape = {segmentation_coalitions.shape}\n' \
        f'Expected: segmentation_coalitions.shape = (n_int*n_coalitions, *data.shape).'
    assert isinstance(data, np.ndarray), 'data needs to be a NDArray.'
    assert isinstance(segmentation_coalitions, np.ndarray), 'segmentation_coalitions needs to be a NDArray.'


def create_clean_imputation(imputations: NDArray, data: NDArray, segmentation_coalitions: NDArray) -> NDArray:
    """Shields imputations and resets all required features to original value."""
    mask_impute = segmentation_coalitions <= -1
    mask_original = np.invert(mask_impute)
    mask_original_broadcast = np.broadcast_to(mask_original[None], imputations.shape)
    x_broadcast = np.broadcast_to(data, imputations.shape)
    imputations[mask_original_broadcast] = x_broadcast[mask_original_broadcast]
    return imputations


class Imputer(ABC):
    imputer_name: str = None

    def impute(self, data: NDArray, segmentation_coalitions: NDArray[int], n_imputations: int) -> NDArray:
        """
        Generate imputations for data.

        Args:
            data: shape: (*shape_input, *shape_x) with non-trivial shape_input, i.e. (1,)
            segmentation_coalitions: feature collections are indicated by positive integers.
                Negative integers feature are to be imputed. (..., *shape_input, *shape_x)
            n_imputations: number of independent imputations

        Returns:
            imputations: (n_imputations,..., *shape_input, *shape_x)
        """
        _check_input(data=data, segmentation_coalitions=segmentation_coalitions)
        imputations_temp = self._impute(data=data, segmentation_coalitions=segmentation_coalitions,
                                        n_imputations=n_imputations)
        assert imputations_temp.shape == (n_imputations, *segmentation_coalitions.shape), \
            'Imputer alters shape of input, expect only an additional leading imputation axis.'
        imputations = create_clean_imputation(imputations=imputations_temp, data=data,
                                              segmentation_coalitions=segmentation_coalitions)
        return imputations

    @abstractmethod
    def _impute(self, data: NDArray, segmentation_coalitions: NDArray[int], n_imputations: int) -> NDArray:
        """To be overwritten by all new imputers to be compatible with the conditional_explainer package."""
