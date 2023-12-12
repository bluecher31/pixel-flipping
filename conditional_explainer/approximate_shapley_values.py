import numpy as np
from scipy.special import binom

from sklearn.linear_model import LinearRegression, Ridge

from typing import Callable, Dict, List, Set
from numpy.typing import NDArray

from conditional_explainer.base_explainer import BaseExplainer
from conditional_explainer.imputers.abstract_imputer import Imputer
from conditional_explainer.shapley import ShapleyValues
from conditional_explainer.helper_kernelshap import shap_kernel_weight
from helper_kernelshap import convert_coalitions_to_segmentation


def shapley_normalization(n_features: int, cardinality: int) -> float:
    """
    normalization for each shapley term
    Args:
        n_features: 'n = |N|'
        cardinality: 's = |S|'

    Returns:
        s! * (n - s - 1)! / n! = 1 / [n * binom(n-1, s)]
    """
    assert n_features > cardinality >= 0
    div = binom(n_features - 1, cardinality) * n_features
    return 1 / div


class ApproximateShapleyValues(BaseExplainer):
    def __init__(self, model_fn: Callable, imputer: Imputer, n_eval: int, data: NDArray,
                 cardinality_coalitions: List[int] = None):
        """
        Initializes an explainer based on KernelShap approach to calculate Shapley values.

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
        """
        super().__init__(model_fn=model_fn, imputer=imputer, n_eval=n_eval, data=data)
        assert len(cardinality_coalitions) < self.n_eval, 'Only at least on evaluation per cardinality.'
        self.shapley = ShapleyValues(model_fn=model_fn, imputer=imputer, n_eval=int(n_eval/len(cardinality_coalitions)),
                                     data=data, cardinality_coalitions=cardinality_coalitions)
        self.cardinality_coalitions = cardinality_coalitions
        self.explainer_name = 'ApproximateShapleyValues'

    def _attribution(self, data: NDArray, segmentation: NDArray, target_features: Set[int]) -> Dict[str, NDArray]:
        """
        All features are used for attributions and therefore target_features is redundant.

         Args:
            data: input samples for which attributions are calculated, shape: (*shape_input, *shape_x),
                usually shape_input = (n_samples) but more dimensions are possible,
            segmentation: feature segmentation for data indicated by positive integers, shape: (*shape_input, *shape_x)
            target_features: redundant, all attributions are calculated.  Needs to consist of all features,
                shape: (n_features)

        """
        self.features = set(segmentation.flatten())
        assert self.features.difference(target_features) == set(), \
            'This method only computes attributions for all features simultaneously'
        self.n_features = len(self.features)
        # self.n_int = 1      # this dimension is not needed this joined feature approach
        assert len(self.shape_input) == 1, 'Please flatten the sample dimensions for your input data.'

        cardinality_positive = [c for c in self.cardinality_coalitions if c > 0]
        cardinality_negative = [self.n_features - abs(c) for c in self.cardinality_coalitions if c <= 0]
        cardinality_coalitions = [*cardinality_positive,
                                  *[c for c in cardinality_negative if c >= 0]]
        # remove redundant cardinalities (too large and double entries)
        cardinality_coalitions = list(set([c for c in cardinality_coalitions if c < self.n_features]))

        dict_attributions = {}
        for target in target_features:
            attr_s_list = []
            weight_s_list = []
            for s in cardinality_coalitions:
                self.shapley.cardinality_coalitions = [s]
                dict_attr_cardinality = \
                    self.shapley.attribution(data=data, segmentation=segmentation, target_features={target})
                attr_s_list.append(dict_attr_cardinality[f'{target}'])
                weight_s_list.append(shapley_normalization(n_features=self.n_features, cardinality=s))
            attr_s = np.stack(attr_s_list)
            # weights = np.stack(weight_s_list)
            # attr = np.einsum('i,i...->...', weights, attr_s)
            n = len(attr_s) + 1
            attr = attr_s.sum(axis=0) / n
            dict_attributions[f'{target}'] = attr

            # dict_attr = super(ApproximateShapleyValues, self)._attribution(
            #     data=data, segmentation=segmentation, target_features={target})
            # dict_attributions.update(dict_attr)

        dict_attributions['true_prediction'] = self.model_fn(data)
        occluded_prediction = self._occluded_model_fn(data=data, segmentation_coalitions=-segmentation[None, None])
        dict_attributions['mean_prediction'] = occluded_prediction.reshape((*self.shape_input, self.n_classes))

        # normalize attributions to fulfil efficiency property
        assert self.n_classes == 1, 'Normalization not implemented for multi-class classification.'
        for i_sample, [prediction, empty_prediction] in \
                enumerate(zip(dict_attributions['true_prediction'], dict_attributions['mean_prediction'])):
            attributions = [float(dict_attributions[f'{feature}'][i_sample])
                            for feature in target_features]
            rescale_factor = float(prediction - empty_prediction) / np.sum(attributions)
            # assert rescale_factor > 0, 'Do not allow for sign switch.'
            for target in target_features:
                dict_attributions[f'{target}'][i_sample] *= rescale_factor

        dict_information = {'target_features': target_features, 'data': data, 'segmentation': segmentation}
        dict_explanation = {**dict_attributions, **dict_information, **self.get_summary()}

        # n_cardinalities = len(self.cardinality_coalitions)
        #
        # predictions_cardinality = np.stack([self.model_cardinality_fn(data=data,
        #                                                               cardinality=cardinality,
        #                                                               target_features=target_features,
        #                                                               segmentation=segmentation)
        #                                     for cardinality in self.cardinality_coalitions])
        # # shape: (n_cardinalities, 2, *shape_input, n_classes)
        #
        # # get weight for each coalition
        # cardinality_weight = np.array([shapley_normalization(n_features=self.n_features, cardinality=cardinality)
        #                                for cardinality in self.cardinality_coalitions])
        # # shape: (n_coalitions)
        #
        # dict_attributions = self._calculate_attributions(occluded_predictions=predictions_cardinality,
        #                                                  coalition_weights=cardinality_weight,)
        # dict_attributions['segmentation'] = segmentation
        return dict_explanation

    # def model_cardinality_fn(self, data: NDArray, target_features: Set[int], segmentation: NDArray, cardinality: int) -> NDArray:
    #     """
    #     Args:
    #         cardinality: size of coalitions which are sampled to calculate average occluded prediction.
    #     Returns:
    #         shape: (2, *shape_input, n_classes)
    #     """
    #     list_coalitions = self._generate_coalitions()
    #     assert len(list_coalitions) == self.n_coalitions, 'Please provide the correct number of coalitions. '
    #     raise NotImplementedError

    # def _generate_coalitions(self) -> List[Set[int]]:
    #     """
    #     Generates random coalition between all features.
    #     The empty (full) coalition are always included at first (second) position.
    #
    #     Returns:
    #         list_coalitions: List[Set], i.e. [empty_set, set(features), S3,..., S_{n_coalitions}]
    #
    #     """
    #     # deterministically generate coalitions
    #     if self.subsample_coalitions is False:      # use all possible coalitions
    #         list_coalitions_raw = helper_kernelshap.power_set_coalitions(
    #             features=self.features, cardinality_coalitions=self.cardinality_coalitions
    #         )
    #         # reset numerical parameters to match new n_coalitions
    #         self.n_imputations = int(self.n_eval/(len(list_coalitions_raw) + 2))
    #         self.n_eval = self.n_imputations * (len(list_coalitions_raw) + 2)
    #     else:           # random subset
    #         list_coalitions_raw = helper_kernelshap.random_coalitions(
    #             features=self.features, n_coalitions=self.n_coalitions-2,
    #             cardinality_coalitions=self.cardinality_coalitions
    #         )
    #     list_coalitions = [set(), self.features, *list_coalitions_raw]
    #     # list_coalitions = list_coalitions_raw
    #     return list_coalitions

    # def _calculate_attributions(self, occluded_predictions: NDArray, coalition_weights: NDArray,
    #                             list_coalitions: List[Set]) -> Dict[str, NDArray]:
    #     """
    #     Fit a weighted linear model and create an appropriate dict with all attributions.
    #
    #     Args:
    #         occluded_predictions: f(S), the remaining features S_bar are occluded with the imputer_fn,
    #             shape: (n_coalitions, *shape_input, n_classes)
    #         coalition_weights: the weights for kernel shap weights for each coalition, shape: (n_coalitions)
    #         list_coalitions: list of coalitions S which are present during the prediction,
    #             [full, empty_set, S3,..., S_{n_coalitions}], len(list_coalitions): n_coalitions
    #
    #     Returns:
    #         dict_attributions
    #     """
    #     assert set() in list_coalitions and self.features in list_coalitions, \
    #         'Empty set and set(features) need to be provided.'
    #     np_features = np.array(sorted(self.features))       # ordering is relevant to assign attributions later
    #
    #     list_feature_coalitions = []
    #     for i_coalition, coalition in enumerate(list_coalitions):
    #         mask = np.zeros(self.n_features, dtype=bool)
    #         for feature in coalition:
    #             mask[np_features == feature] = True
    #         list_feature_coalitions.append(mask)
    #     coalitions_masked_on_features = np.stack(list_feature_coalitions)      # binary representation of all coalitions
    #
    #     list_attributions = []
    #     for i_sample in range(self.shape_input[0]):  # loop all samples and fit each data point individually
    #         occluded_prediction_sample = occluded_predictions[0, :, i_sample]
    #
    #         # fit weighted linear model
    #         if self.ridge_parameter == 0:
    #             regr = LinearRegression()
    #         elif self.ridge_parameter > 0:
    #             regr = Ridge(alpha=self.ridge_parameter)
    #         else:
    #             raise ValueError(f'Invalid ridge_parameter: {self.ridge_parameter}')
    #
    #         # regr.fit(X, y, weights)
    #         regr.fit(coalitions_masked_on_features, occluded_prediction_sample, np.squeeze(coalition_weights))
    #         # regr.fit(coalitions_masked_on_features[2:], occluded_prediction_sample[2:],
    #         #          np.squeeze(coalition_weights[2:]))
    #
    #         phi = regr.coef_
    #         list_attributions.append(np.moveaxis(phi, 0, -1))
    #     attributions = np.stack(list_attributions)
    #
    #     # convert attributions into standard format
    #     dict_attributions = {}
    #     for i, feature in enumerate(np_features):
    #         dict_attributions[f'{feature}'] = attributions[:, i]
    #     for i_coalition, coalition in enumerate(list_coalitions):
    #         if coalition == set(self.features):
    #             dict_attributions['true_prediction'] = occluded_predictions[0, i_coalition]
    #         elif coalition == set({}):
    #             dict_attributions['mean_prediction'] = occluded_predictions[0, i_coalition]
    #     return dict_attributions
