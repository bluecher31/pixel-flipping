import numpy as np


from conditional_explainer.imputers.abstract_imputer import create_clean_imputation

from conditional_explainer.imputers.abstract_imputer import Imputer
from numpy.typing import NDArray


class IdentityImputer(Imputer):
    imputer_name = 'IdentityImputer'

    def _impute(self, data: NDArray, segmentation_coalitions: NDArray, n_imputations: int) -> NDArray:
        """
        Impute with identical feature values
        Args:
            data: shape: (*shape_input, shape_x)
            segmentation_coalitions: shape: (n_masks, *shape_input, *shape_x)
            n_imputations: requested number of different imputations

        Returns:
            zeros: shape: (n_masks, *shape_input, *shape_x)
        """
        n_mask = segmentation_coalitions.shape[0]            # this axis needs to be modified for recycle_imputation == True

        data_broadcast = np.broadcast_to(data, shape=(n_imputations, n_mask, *segmentation_coalitions.shape[1:]))
        imputations = np.array(data_broadcast)        # make writeable
        return imputations


class ConstantValueImputer(Imputer):
    imputer_name = 'ConstantValueImputer'

    def __init__(self, constant: float):
        self.constant = constant

    def _impute(self, data: NDArray, segmentation_coalitions: NDArray, n_imputations: int) \
            -> NDArray:
        """
        Impute constant everywhere
        Args:
            data: shape: (*shape_input, shape_x)
            segmentation_coalitions: shape: (n_masks, *shape_input, *shape_x)
            n_imputations: requested number of different imputations
        Returns:
            imputations: shape: (n_masks, *shape_input, *shape_x)
        """
        # _check_input(data, segmentation_coalitions)

        n_mask = segmentation_coalitions.shape[0]
        imputations = self.constant * np.ones((n_imputations, n_mask, *data.shape))
        imputations = create_clean_imputation(imputations=imputations, data=data,
                                              segmentation_coalitions=segmentation_coalitions)
        return imputations


class TrainSetImputer(Imputer):
    """
    imputer just inserts randomly sampled training samples
    """
    def __init__(self, train_data: NDArray):
        super().__init__()
        self.imputer_name = 'TrainSet'

        self.train_data = train_data.copy()
        self.shape_x = self.train_data.shape[1:]
        self.n_train = len(self.train_data)
        print(f'TrainSet imputer: \n'
              f'shape_x = {self.shape_x}\n'
              f'n_train = {self.n_train}')
        self.seg = None

    def _impute(self, data: NDArray, segmentation_coalitions: NDArray, n_imputations: int) -> NDArray:
        """
        Args:
            data: shape: (*shape_input, shape_x)
            segmentation_coalitions: shape: (n_masks, *shape_input, *shape_x)
            n_imputations: requested number of different imputations
        Returns:
            imputations: shape: (n_masks, *shape_input, *shape_x)"""
        # returns only imputation for a single mask
        # _check_input(data, segmentation_coalitions)

        np_test = np.array(data)
        n_samples = len(np_test.reshape((-1, *self.shape_x)))
        n_mask = segmentation_coalitions.shape[0]  # this axis needs to be modified for recycle_imputation == True
        rng = np.random.default_rng()
        # rng.choice(self.train_data, n_samples, replace=True)  # replace: multiple occurrences are allowed
        res = rng.choice(self.train_data, n_samples * n_imputations * n_mask, replace=True).copy()
        imputations = res.reshape((n_imputations, n_mask, *data.shape))

        # imputations = create_clean_imputation(imputations=imputations, data=data,
        #                                       segmentation_coalitions=segmentation_coalitions)
        return imputations


class GaussianNoiseImputer(Imputer):
    imputer_name = 'GaussianNoiseImputer'
    def __init__(self, mu_data: NDArray, cov_data: NDArray):
        """Creates a gaussian imputer according to N(mu_data, cov_data) and returns it as a Callable fn.
          Args:
              mu_data: mean normal distribution
              cov_data: covariance
          Returns:
              object: Callable imputer_fn
          """
        self.rng = np.random.default_rng(seed=1)
        self.mu_data = mu_data
        self.cov_data = cov_data
        n_features = len(mu_data)

        # TODO: Implemented correlated gaussian variable imputer
        if np.alltrue(cov_data == np.eye(n_features)) is False:
            raise NotImplementedError('Not implemented for correlated variables.')

    def _impute(self, data: NDArray, segmentation_coalitions: NDArray[int], n_imputations: int) -> NDArray:
        """
            Args:
                data: shape: (*shape_input, shape_x)
                segmentation_coalitions: shape: (n_masks, *shape_input, *shape_x)
                n_imputations: requested number of different imputations
            Returns:
                zeros: (n_int, n_imputations, n_S, *shape_input)"""
        n_samples = data[..., 0].size
        n_mask = segmentation_coalitions.shape[0]  # this axis needs to be modified for recycle_imputation == True

        imputations_temp = self.rng.multivariate_normal(mean=self.mu_data, cov=self.cov_data,
                                                   size=n_imputations * n_mask * n_samples)

        imputations = imputations_temp.reshape((n_imputations, n_mask, *data.shape))

        imputations = create_clean_imputation(imputations=imputations, data=data,
                                              segmentation_coalitions=segmentation_coalitions)

        return imputations
