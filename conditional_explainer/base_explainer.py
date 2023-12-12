import numpy as np

from functools import partial
from typing import Callable, Dict, Tuple, Set

from numpy.typing import NDArray

from conditional_explainer.imputers.abstract_imputer import Imputer
from conditional_explainer import helper_base

from abc import ABC, abstractmethod

# Data: NDArray        # (*shape_input, *shape_x)
# Output = Annotated[NDArray, '(*shape_input, n_classes)']                      # (*shape_input, n_classes)
# Segmentation = NDArray[int]               # Data.shape, dtype=int
# TargetFeatures = NDArray             # dtype=int, np.unique(TargetFeatures) == TargetFeatures
#
# SegmentationCoalition = NDArray      # (n_int, n_coalitions, *shape_input, *shape_x)
# OccludedPredictions = SegmentationCoalition
# Imputations = NDArray                # (n_int, n_imputations, n_coalitions, *shape_input, *shape_x)

ModelFn = Callable[[NDArray], NDArray]

DictAttributions = Dict[str, NDArray]            # target_features = [1, 2] -> keys = ['1', '2', '1^2']


class BaseExplainer(ABC):
    def __init__(self,
                 model_fn: ModelFn,
                 imputer: Imputer,
                 n_eval: int,
                 data: NDArray):
        """
        Abstract explainer base class.

        Args:
            model_fn: callable model, shapes: (*shape_input, *shape_x) -> (*shape_input, n_classes),
            imputer: models conditional distribution which is used to occlude features and build an occluded_model_fn.
                function signature: [data: NDArray, segmentation_coalition: NDArray, n_imputations: int] -> imputations,
                data.shape: (*shape_input, *shape_x)
                segmentation_coalition.shape = (n_masks, *shape_input, *shape_x)
                imputations.shape = (n_imputations, n_masks, *shape_input, *shape_x)
            n_eval: approximate number of model calls
            data: arbitrary test samples, required to probe model_fn/impute_fn for clf vs. reg and shapes,
                shape: (*shape_input, *shape_x)
        """
        self.explainer_name = ''
        # user input
        self.model_fn = model_fn
        self.imputer = imputer
        self.n_eval = n_eval        # approximate number of model_fn calls
        self.n_imputations = 1      # exact number of monte carlo samples to estimate cond. integral

        # maximum number of interacting features, if not -> interaction_depth=len(target_features)
        self.interaction_depth: int = -1            # TODO: how to access from outside?
        self.batch_size = 512           # for manually batching inference for model_fn # TODO: provide access from outside

        (self.shape_x, self.n_classes) = helper_base.get_and_check_shapes(model_fn=self.model_fn, data=data)
        # define link_fn for occluded_predictionsw
        if self.n_classes == 1:         # regression setting
            self.transform_occluded_predictions_fn = lambda y: y
        else:
            # smoothing probabilities to avoid zero division
            laplace_smoothing_fn = partial(helper_base.laplace_smoothing, n_train=10_000, n_classes=self.n_classes)
            self.transform_occluded_predictions_fn = lambda prob: np.log2(laplace_smoothing_fn(prob))

        single_sample = data.reshape(-1, *self.shape_x)[0]
        helper_base.partial_check_impute_fn(impute_fn=self.imputer.impute, data=single_sample)

        self.shape_input: Tuple = ()     # is assigned on-the-fly during attribution() call
        self.n_int: Tuple = ()  # is assigned on-the-fly during attribution() call

    def attribution(self, data: NDArray, segmentation: NDArray, target_features: Set[int]) -> Dict[str, NDArray]:
        """
        Calculates attributions between the target_features.
        Refer to derive _attribution method for method-specific documentation.

        Args:
            data: input samples for which attributions are calculated, shape: (*shape_input, *shape_x),
                usually shape_input = (n_samples) but more dimensions are possible,
            segmentation: feature segmentation for data indicated by positive integers, shape: (*shape_input, *shape_x)
            target_features: set of 'n_targets' positive integers, i.e. [1, 2, 3], len(target_features) = n_targets

        Returns:
            Dictionary with all calculate n-point attributions.
            Consider requested target_features = [1, 2, 3] and all possible 3-point interactions:

            {'1': NDArray, '2': NDArray, '3': NDArray,
             '1^2': NDArray, '1^3': NDArray, '2^3': NDArray,
             '1^2^3': NDArray}

             keys are strings, interacting features are separated via a '^'.
        """
        self.shape_input = data.shape[:-len(self.shape_x)]

        # # check compatibility of inputs
        # if len(self.shape_input) == 0:          # add dimension for single sample
        #     data = data[np.newaxis]
        #     segmentation = segmentation[np.newaxis]
        #     self.shape_input = (1,)

        # TODO: remove this limitation and allow for single flat samples (keeping shape of segmentation, data consistent)
        assert len(self.shape_input) > 0, 'at least one sample dimension'
        assert data.shape == (*self.shape_input, *self.shape_x), 'Incorrect data shape'
        assert segmentation.shape == data.shape, 'Segmentation does not fit to data shape'
        assert np.alltrue(segmentation > 0), 'Only positive feature labels allowed.'
        assert segmentation.dtype == np.int, 'Non-integer segmentation.'
        assert isinstance(target_features, set), 'target_features needs to be a set of positive integers.'

        features_per_sample = [set(sample_segmentation.flatten())
                               for sample_segmentation in segmentation.reshape((-1, *self.shape_x))]
        assert len(np.unique(features_per_sample)), f'All samples should have the same set of features. \n' \
                                                    f'{np.unique(features_per_sample)}'

        set_features = set(segmentation.flatten())
        assert target_features.issubset(set_features), \
            f'target_features = {target_features} is not a a subset of ' \
            f'set(segmentation) = {set(segmentation.flatten())}'
        assert len(set(target_features)) == len(target_features), 'target_features not unique'

        dict_attributions_tmp = self._attribution(data=data, segmentation=segmentation, target_features=target_features)
        # ensure that interactions keys are sorted in increasing order ('2^4^1^3' -> '1^2^3^4')

        dict_attributions = {helper_base.organize_key(key): dict_attributions_tmp[key] for key in dict_attributions_tmp}

        return dict_attributions

    @abstractmethod
    def _attribution(self, data: NDArray, segmentation: NDArray, target_features: Set[int]) -> Dict[str, NDArray]:
        """Please refer to public method."""

    def external_occluded_model_fn(self, data: NDArray, segmentation_coalitions: NDArray) \
            -> NDArray:
        """
          Returns occluded prediction using impute_fn.

          Args:
              data: (*shape_input, *shape_x), usually shape_input = (n_samples) but more dimensions are possible
              segmentation_coalitions: (n_int, n_S, *shape_input, *shape_x)
          Returns:
              occluded predictions, shape: (n_int, n_coalitions, shape_input, n_classes)

          Notes:
              This function can be overwritten to directly supply an occluded_model_fn externally.

          """
        # TODO: handle case if sample is not imputed at all imputations = x
        # # cut out trivial segmentations from segmentation_coalition
        # mask = (segmentation_coalitions > 0)  # true if feature are not imputed
        # # shape: (n_int, n_S, *shape_input, *shape_x)
        #
        # # flatten sample dimension and check whether all features are untouched
        # mask_identity = np.alltrue(mask.reshape(*mask.shape[:-len(self.shape_x)], -1), axis=-1)
        # # shape: (n_int, n_S, *shape_input)
        # mask_imputations = np.all(mask_identity.reshape(*mask_identity.shape[:-len(self.shape_input)], -1), axis=-1)
        # # shape: (n_int, n_S)

        # TODO: this function will assume data.shape == segmentation.shape, with flat input dimension
        #  the broadcasted data will be ignored afterwards
        imputations = self._impute(data, segmentation_coalitions, self.n_imputations)
        # shape: (n_imputations, n_int, n_coalitions, *input_shape, *shape_x)

        imputed_predictions = self._model_fn(data=imputations)
        # shape: (n_imputations, n_int, n_coalitions, *input_shape, n_classes)
        # # insert true prediction here

        occluded_prediction = np.mean(imputed_predictions, axis=0)
        # shape: (n_int, n_coalitions, *input_shape, n_classes)

        return occluded_prediction

    def _occluded_model_fn(self, data: NDArray, segmentation_coalitions: NDArray) \
            -> NDArray:
        """
        Returns occluded prediction rescaled by the specified transformation (e.g. identity or log2).

        Args:
            data: (*shape_input, *shape_x), usually shape_input = (n_samples) but more dimensions are possible
            segmentation_coalitions: (n_int, n_coalitions, *shape_input, *shape_x)
        Returns:
            occluded predictions, shape: (n_int, n_coalitions, shape_input, n_classes)
        """
        # TODO: this function will broadcast data to achieve data.shape == segmentation.shape, flattened
        occluded_prediction = self.external_occluded_model_fn(data, segmentation_coalitions)


        return self.transform_occluded_predictions_fn(occluded_prediction)

    def _impute(self, data: NDArray, segmentation_coalitions: NDArray, n_imputations: int) -> NDArray:
        """
        Wrapper for public impute_fn.
        Provides different possibilities to change the imputation signature as unified_integral and
        recycling of imputations. This is in particular interesting for the interaction axis (n_int).

        Args:
            data: to be occluded, shape: (*shape_input, *shape_x)
            segmentation_coalitions: (n_int, n_coalitions, *shape_input, *shape_x)
            n_imputations: number of different imputations

        Returns:
            imputations, shape: (n_imputations, n_int, n_coalitions, *shape_input, *shape_x)
        """
        # TODO: implement recycle_imputation/unified_integral functionality within this wrapper
        #  introduce a keyword to access different options

        shape_flat = (-1, *self.shape_input, *self.shape_x)
        segmentation_flat = segmentation_coalitions.reshape(shape_flat)

        imputations_flat = self.imputer.impute(data, segmentation_flat, n_imputations)

        shape_imputation = (n_imputations, *segmentation_coalitions.shape)
        imputations = imputations_flat.reshape(shape_imputation)
        return imputations

    def _model_fn(self, data: NDArray) -> NDArray:
        """
        Wrapper to handle internal dimensional requirements.

        Args:
            data: usually imputations, shape: (..., shape_x)

        Returns: predictions, shape: (n_imputations, n_int, n_coalitions, *shape_input, n_classes)
        """
        input_shape = data.shape[:-len(self.shape_x)]
        data_flat = data.reshape((-1, *self.shape_x))
        predictions_flat = np.concatenate([np.array(self.model_fn(data_flat[i*self.batch_size:(i+1)*self.batch_size]))
                                           for i in range(int(len(data_flat) / self.batch_size) + 1)],
                                          axis=0)
        predictions_reshaped = predictions_flat.reshape((*input_shape, self.n_classes))
        # shape: (n_imputations, n_int, n_coalitions, *shape_input, n_classes)
        return predictions_reshaped

    @property
    def n_coalitions(self):
        """
        Returns the number of coalitions.
        Sometimes also denoted as n_S,
        This dimension can be used to estimate confidence intervals for attributions.
        """
        assert self.n_eval >= self.n_imputations >= 1
        n = int(self.n_eval / self.n_imputations)
        assert self.n_eval >= n >= 1
        return n

    def get_summary(self) -> Dict:
        """Returns all relevant information of the current explainer object."""
        # TODO: convert into __str__ or __repr__
        dict_explainer_info = {
            'explainer_name': self.explainer_name,
            'model_fn': str(self.model_fn),
            'imputer': self.imputer.imputer_name,
            'interaction_depth': self.interaction_depth,
            'n_eval': self.n_eval,
            'n_imputations': self.n_imputations,
            'shape_x': self.shape_x,
            'n_classes': self.n_classes
        }
        return dict_explainer_info

#
# def create_occluded_model_explainer(explainer_cls: BaseExplainer, occluded_model: Callable, n_eval: int, data: NDArray) \
#         -> BaseExplainer:
#     """
#     Overwrites the _occluded_model method of the BaseExplainer class with an externally provided occluded_model_fn.
#     """
#     from conditional_explainer.imputers.simple_imputers import IdentityImputer
#     dummy_imputer = IdentityImputer()
#     explainer = explainer_cls.__init__(model_fn=occluded_model, imputer=dummy_imputer, n_eval=n_eval, data=data)
#     def internal_occluded_prediction()
#     raise NotImplementedError()
#     return explainer
#
