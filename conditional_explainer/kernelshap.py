import numpy as np

from sklearn.linear_model import LinearRegression, Ridge

from typing import Callable, Dict, List, Set
from numpy.typing import NDArray

from conditional_explainer.base_explainer import BaseExplainer
from conditional_explainer.imputers.abstract_imputer import Imputer
from conditional_explainer.helper_kernelshap import shap_kernel_weight
from conditional_explainer.helper_kernelshap import convert_coalitions_to_segmentation
from conditional_explainer import helper_kernelshap


class KernelSHAP(BaseExplainer):
    """
    https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/model_agnostic/Simple%20Kernel%20SHAP.html
    """
    def __init__(self, model_fn: Callable, imputer: Imputer, n_eval: int, data: NDArray,
                 cardinality_coalitions: List[int] = None, subsample_coalitions: bool = True,
                 ridge_parameter: float = 0., symmetric_coalitions: bool = False):
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
            subsample_coalitions: randomly generate exactly n_coalitions coalitions (using prior_cardinality='binom'),
                if False all possible coalitions will be calculated (this might be a very large number).
            ridge_parameter: regularize the magnitude of shapley values.
            symmetric_coalitions: always complementary coalitions to improve convergence as
                proposed in Covert&Lee: Improved KernelSHAP @ AISTATS 21


        """
        super().__init__(model_fn=model_fn, imputer=imputer, n_eval=n_eval, data=data)
        self.n_imputations = 1
        self.explainer_name = 'KernelSHAP'

        self.cardinality_coalitions = cardinality_coalitions
        self.subsample_coalitions = subsample_coalitions
        self.ridge_parameter = ridge_parameter
        self.symmetric_coalitions = symmetric_coalitions

        self.rng = np.random.default_rng(seed=0)

    def _attribution(self, data: NDArray, segmentation: NDArray, target_features: Set[int]) -> Dict[str, NDArray]:
        """
        See Also: parent method documentation.
        Calculates all single (1-point) feature attributions based on the KernelSHAP algorithm.
        http://proceedings.mlr.press/v130/covert21a/covert21a.pdf Improved KernelSHAP

        All features are used for attributions and therefore target_features is redundant.

         Args:
            data: input samples for which attributions are calculated, shape: (*shape_input, *shape_x),
                usually shape_input = (n_samples) but more dimensions are possible,
            segmentation: feature segmentation for data indicated by positive integers, shape: (*shape_input, *shape_x)
            target_features: redundant, all attributions are calculated.  Needs to consist of all features,
                shape: (n_features)

        """
        # TODO: check whether this really generalizes to more samples at once, if segmentation is not consistent

        # TODO: apparently there are two option: (reference needed)
        #  (i) sample S according to shapley weight and fit uniform linear model
        #  (ii) sample S uniformly and fit shapley weighted linear model

        # TODO: check captum implementation or https://docs.seldon.io/projects/alibi/en/stable/methods/KernelSHAP.html
        self.features = set(segmentation.flatten())
        assert self.features.difference(target_features) == set(), \
            'This method only computes attributions for all features simultaneously'
        self.n_features = len(self.features)
        self.n_int = 1      # this dimension is not needed this joined feature approach
        assert len(self.shape_input) == 1, 'Please flatten the sample dimensions for your input data.'

        list_coalitions = self._generate_coalitions()
        assert len(list_coalitions) == self.n_coalitions, 'Please provide the correct number of coalitions. '

        # compute occluded predictions
        segmentation_coalitions = convert_coalitions_to_segmentation(segmentation, list_coalitions)
        occluded_predictions = self._occluded_model_fn(data, segmentation_coalitions)
        # shape: (1, n_coalitions, *shape_input, n_classes)

        # get weight for each coalition
        coalition_weights = np.array([shap_kernel_weight(n_features=self.n_features, size_coalition=len(coalition))
                                      for coalition in list_coalitions])    # shape: (n_coalitions)

        dict_attributions = self._fit_weighted_linear_model(occluded_predictions=occluded_predictions,
                                                            coalition_weights=coalition_weights,
                                                            list_coalitions=list_coalitions)
        dict_attributions['segmentation'] = segmentation
        return dict_attributions

    def _generate_coalitions(self) -> List[Set[int]]:
        """
        Generates random coalition between all features.
        The empty (full) coalition are always included at first (second) position.

        Returns:
            list_coalitions: List[Set], i.e. [empty_set, set(features), S3,..., S_{n_coalitions}]

        """
        # deterministically generate coalitions
        if self.subsample_coalitions is False:      # use all possible coalitions
            list_coalitions_raw = helper_kernelshap.power_set_coalitions(
                features=self.features, cardinality_coalitions=self.cardinality_coalitions
            )
            # reset numerical parameters to match new n_coalitions
            self.n_imputations = int(self.n_eval/(len(list_coalitions_raw) + 2))
            self.n_eval = self.n_imputations * (len(list_coalitions_raw) + 2)
        else:           # random subset
            list_coalitions_raw = helper_kernelshap.random_coalitions(
                features=self.features, n_coalitions=self.n_coalitions-2,
                cardinality_coalitions=self.cardinality_coalitions, symmetric_coalitions=self.symmetric_coalitions
            )
        list_coalitions = [set(), self.features, *list_coalitions_raw]
        # list_coalitions = list_coalitions_raw
        return list_coalitions

    def _fit_weighted_linear_model(self, occluded_predictions: NDArray, coalition_weights: NDArray,
                                   list_coalitions: List[Set]) -> Dict[str, NDArray]:
        """
        Fit a weighted linear model and create an appropriate dict with all attributions.

        Args:
            occluded_predictions: f(S), the remaining features S_bar are occluded with the imputer_fn,
                shape: (n_coalitions, *shape_input, n_classes)
            coalition_weights: the weights for kernel shap weights for each coalition, shape: (n_coalitions)
            list_coalitions: list of coalitions S which are present during the prediction,
                [full, empty_set, S3,..., S_{n_coalitions}], len(list_coalitions): n_coalitions

        Returns:
            dict_attributions
        """
        assert set() in list_coalitions and self.features in list_coalitions, \
            'Empty set and set(features) need to be provided.'
        np_features = np.array(sorted(self.features))       # ordering is relevant to assign attributions later

        list_feature_coalitions = []
        for i_coalition, coalition in enumerate(list_coalitions):
            mask = np.zeros(self.n_features, dtype=bool)
            for feature in coalition:
                mask[np_features == feature] = True
            list_feature_coalitions.append(mask)
        coalitions_masked_on_features = np.stack(list_feature_coalitions)      # binary representation of all coalitions

        list_attributions = []
        for i_sample in range(self.shape_input[0]):  # loop all samples and fit each data point individually
            occluded_prediction_sample = occluded_predictions[0, :, i_sample]

            # fit weighted linear model
            if self.ridge_parameter == 0:
                regr = LinearRegression()
            elif self.ridge_parameter > 0:
                regr = Ridge(alpha=self.ridge_parameter)
            else:
                raise ValueError(f'Invalid ridge_parameter: {self.ridge_parameter}')

            # regr.fit(X, y, weights)
            regr.fit(coalitions_masked_on_features, occluded_prediction_sample, np.squeeze(coalition_weights))
            # regr.fit(coalitions_masked_on_features[2:], occluded_prediction_sample[2:],
            #          np.squeeze(coalition_weights[2:]))

            phi = regr.coef_
            list_attributions.append(np.moveaxis(phi, 0, -1))
        attributions = np.stack(list_attributions)

        # convert attributions into standard format
        dict_attributions = {}
        for i, feature in enumerate(np_features):
            dict_attributions[f'{feature}'] = attributions[:, i]
        for i_coalition, coalition in enumerate(list_coalitions):
            if coalition == set(self.features):
                dict_attributions['true_prediction'] = occluded_predictions[0, i_coalition]
            elif coalition == set({}):
                dict_attributions['mean_prediction'] = occluded_predictions[0, i_coalition]
        return dict_attributions
