import torch

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import interpolate, integrate

from src.datasets.imagenet import preprocess_imagenet
from src.experiments.attribution import calculate_similarity

from src.interface import Image, Measurement
from src.experiments.attribution import Attribution, load_attributions, root_attribution
from src.experiments.pixel_flipping import PixelFlipping
from typing import List
from numpy.typing import NDArray

from src.config.helpers import load_all_configs, filter_config_folders
from src.config_matplotlib import line_with_shaded_errorband
from copy import deepcopy


def imshow_attribution(ax: plt.Axes, digit: NDArray, heatmap: NDArray):
    """Imshow digit with superimposed heatmap."""
    ax.imshow(digit, cmap=plt.cm.binary)
    # argsort = np.argsort(heatmap)
    vmax = np.max(np.abs(heatmap))
    # ax.imshow(argsort, cmap='gist_heat', alpha=0.75)
    im = ax.imshow(heatmap, cmap=plt.cm.coolwarm, vmax=vmax, vmin=-vmax, alpha=0.75)

    # ax.set_axis_off()         # remove frame, ticks and labels

    # leaves frame and removes ticks and labels
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)


def add_text(text: str, ax: plt.Axes, loc: str, fontsize='small'):
    anchored_text = plt.matplotlib.offsetbox.AnchoredText(text, prop=dict(size=fontsize), loc=loc, borderpad=0.03)
    anchored_text.patch.set_boxstyle("round, pad=-0.2, rounding_size=0.1")
    anchored_text.patch.set_alpha(0.8)
    ax.add_artist(anchored_text)


def summarize_rgb_channels(heatmap_rgb: NDArray, which: str, channel_axis: int = 0) -> NDArray:
    if which == 'mean':
        heatmap_summary = np.mean(heatmap_rgb, channel_axis)
    elif which == 'abs_sum':
        heatmap_summary = np.sum(np.abs(heatmap_rgb), channel_axis)
    else:
        raise ValueError(f'Please enter a valid option: {which}')
    return heatmap_summary


def calculate_auc(x: NDArray, y: NDArray, xmin: float, xmax: float) -> float:
    f = interpolate.interp1d(x, y)
    res, err = integrate.quad(f, a=xmin, b=xmax)
    return res


def visualize_attribution(images: List[Image], attributions: List[Attribution], title=''):
    if title == '':
        title = f'attributions - {attributions[0].explainer}'
    else:
        title = 'attributions ' + title

    n_samples = len(images)
    fig = plt.figure(title, figsize=(2 * n_samples, 1.5))
    for i, [image, attribution] in enumerate(zip(images, attributions)):
        assert image.image_name == attribution.image_name, 'Attribution does not match image.'
        ax = fig.add_subplot(1, n_samples, i + 1)
        ax.set_title(f'{image.image_name}\nlabel: {image.label}, model: {int(attribution.prediction.argmax(-1))}')
        if image.dataset == 'imagenet':
            image_np = preprocess_imagenet(image=image)
            heatmap = summarize_rgb_channels(heatmap_rgb=attribution.heatmap, which='abs_sum', channel_axis=0)
        else:
            image_np = image.image
            heatmap = attribution.heatmap
        imshow_attribution(ax=ax, digit=image_np, heatmap=heatmap)
    plt.tight_layout(pad=0.1)


def summarize_pixel_flipping_results(pf_result: List[PixelFlipping]) -> [Measurement, Measurement]:
    occluded_prediction = np.stack([pf.occluded_prediction for pf in pf_result])
    percentage_deleted = np.stack([pf.percentage_deleted for pf in pf_result])
    x_mean = percentage_deleted.mean(0)
    argsort = np.argsort(x_mean)
    x = Measurement(values=percentage_deleted.mean(0)[argsort],
                    error=percentage_deleted.std(0)[argsort]/np.sqrt(percentage_deleted.size))
    y = Measurement(values=occluded_prediction.mean(0)[argsort],
                    error=occluded_prediction.std(0)[argsort] / np.sqrt(occluded_prediction.size))
    return x, y


def compare_pixel_flipping(pixel_flipping_experiments: List[List[PixelFlipping]]):
    fig = plt.figure('pixel flipping')
    ax = fig.add_subplot()
    for pf_results in pixel_flipping_experiments:
        x, y = summarize_pixel_flipping_results(pf_results)
        ax.errorbar(x=x.values, y=y.values, xerr=x.error, yerr=y.error,
                    label=pf_results[0].params.attributions.explainer)

    ax.legend()
    ax.set_title(f'imputer: {pf_results[0].params.imputer_name}')
    ax.set_xlabel('percentage occluded')
    ax.set_ylabel('occluded prediction')
    plt.tight_layout(pad=0.1)


def get_similarities(list_attributions: List[List[Attribution]], references: List[Attribution], which: str) \
        -> [NDArray, NDArray]:
    """Calculates similarities between all provided attributions with respect to the reference attributions."""
    numerical_fidelity = []
    measured_similarity = []
    measured_similarity_error = []

    for attributions in list_attributions:
        all_n_eval = np.stack([int(attr.explainer_properties['n_eval']) for attr in attributions])
        n_eval = int(np.unique(all_n_eval))     # throws an error in case of inconsistency
        numerical_fidelity.append(n_eval)

        similarity, error = calculate_similarity(attributions1=attributions, attributions2=references, which=which)
        measured_similarity.append(similarity)
        measured_similarity_error.append(error)
    return np.array(numerical_fidelity), np.array(measured_similarity), np.array(measured_similarity_error)


def map_method_name(method: str, what: str) -> str:
    if method == 'random':
        color = 'grey'
        name = 'Random'
    elif method == 'gradients':
        color = 'C0'
        name = 'Gradients'
    elif method == 'gradients_nt':
        color = '#94AAD0'
        name = 'NT-Gradients'
    elif method == 'ig' or method == 'ig_abs':
        color = 'C1'
        name = 'IG'
    elif method == 'zennit':
        color = 'C2'
        name = 'LRP'
    elif method == 'ig_nt':
        color = '#99CBA5'
        name = 'NT-IG'
    elif 'input_x_gradients' in method:
        color = 'C5'
        name = 'InputXGradients'
    else:
        color = 'red'
        name = 'default'
    # raise ValueError(f'This method is not defined: {method}')
    if '_abs' in method:
        name = f'abs({name})'

    if what == 'name':
        return name
    elif what == 'color':
        return color
    else:
        raise ValueError(f'This type (what={what}) is not defined. Choose among (name, color)')


def map_imputer_name(name: str, what: str) -> str:
    if name == 'ColorHistogram' or name == 'color_histogram':
        name = 'Histogram'
        color = 'C3'
    elif name == 'ConstantValueImputer' or name == 'constant_value':
        name = 'Mean'
        color = 'C0'
    elif name == 'TrainSet' or name == 'trainset':
        name = 'Train set'
        color = 'C1'
    elif name == 'diffusion':
        name = 'Diffusion'
        color = 'C2'
    elif name == 'cv2':
        color = 'C4'
    elif name == 'internal':
        color = 'C5'
        name = 'Internal'
    else:
        raise ValueError(f'This imputer = {name} is not defined.')

    if what == 'name':
        return name
    elif what == 'color':
        return color
    else:
        raise ValueError(f'This type (what={what}) is not defined. Choose among (name, color)')


def map_attribution_to_name(attr: Attribution, depth: int) -> str:
    """
    Defines a unique name based on the attribution properties. depth: [1, 2,...].
    Special formats are accessible via negative integers:
    -1: formatted \phi^s
    -2: imputer name
    """
    assert depth > 0 or depth in [-1, -2]
    name = ''
    if depth > 0:
        name += str(attr.explainer)

    if depth > 1:
        name += '\n'
        if attr.explainer in ['KernelSHAP', 'Shapley values']:
            name += f'S={attr.explainer_properties["cardinality_coalitions"]}'

    if depth > 2:
        name += '\n'
        if attr.explainer in ['KernelSHAP', 'Shapley values']:
            name += f'symmetric={attr.explainer_properties["symmetric_coalitions"]}'

    if depth == -1:
        if attr.explainer in ['Shapley values']:
            s = attr.explainer_properties["cardinality_coalitions"]
            if s is None:
                name = f'$\phi$'
            elif len(s) == 1:
                s = int(s[0])
                if s >= 0:
                    name = f'$\phi^{{{s}}}$'
                else:
                    name = f'$\phi^{{n-{-s}}}$'
            else:
                raise NotImplementedError
            # assert len(s_temp) == 1 or s_temp is None
            # s = s_temp[0]
            # name += f'$\phi^{{s={s}}}$'
        else:
            raise NotImplementedError(f'No custom description defined for this explainer: {attr.explainer}')
    elif depth == -2:
        name = attr.explainer_properties.get('imputer').get('name')
    return name


def get_explainer_name(attributions: List[Attribution], depth: int = 2) -> str:
    explainer_name_np = np.unique([map_attribution_to_name(attr, depth) for attr in attributions])
    if len(explainer_name_np) != 1:
        raise RuntimeError(f'Inconsistent explainer for provided attributions: \n'
                           f'{len(explainer_name_np)}\n'
                           f'{explainer_name_np}')
    explainer = str(explainer_name_np[0])
    return explainer


def get_explainer_correlation(list_attributions: List[List[Attribution]], depth_explainer_name: int, measure: str) \
        -> pd.DataFrame:
    """
    Calculate all cross correlations/similarities between all provided attributions.

    Args:
        list_attributions: attributions for different explainers, len(list_attributions) = n_explainers
        depth_explainer_name: detail depth (# identifier) for explainer type to distinguish between different explainers
            from the same class (such as Shapley values with s=1 or s=-3)
        measure: which measure should be used to calculate the correlation between the different explainers
            options: ['cosine similarity', 'pearson', 'spearman']

    Returns:
        cross correlation between all explainers. shape: (n_explainers, n_explainers)


    """
    n_explainers = len(list_attributions)
    list_explainers = [get_explainer_name(attrs, depth=depth_explainer_name) for attrs in list_attributions]

    correlation_matrix = np.zeros(shape=(n_explainers, n_explainers))
    for i1, attrs1 in enumerate(list_attributions):
        for i2, attrs2 in enumerate(list_attributions):
            similarity = calculate_similarity(attributions1=attrs1, attributions2=attrs2, which=measure)
            correlation_matrix[i1, i2] = similarity

    df_correlation_matrix = pd.DataFrame(correlation_matrix, columns=list_explainers, index=list_explainers)
    return df_correlation_matrix


def plot_similarity(ax: plt.Axes, numerical_fidelity, similarity, similarity_error,
                    label: str, **kwargs):

    line2d = line_with_shaded_errorband(ax, x=np.array(numerical_fidelity), y=similarity, yerr=similarity_error,
                                        label=label, **kwargs)
    plt.xlabel('model evaluations')
    plt.ylabel('similarity')
    plt.tight_layout(pad=0.1)


def main_correlation_to_shapley_baseline(cfg, ax: plt.Axes):
    n_eval = max(cfg.n_evals)
    all_config_folders = load_all_configs(root=root_attribution(cfg))
    config_folders = filter_config_folders(all_config_folders, key='explainer.n_eval', value=n_eval)

    cfg.explainer.n_eval = n_eval
    cfg.explainer.cardinality_coalitions = None
    attributions_shapley_baseline = load_attributions(cfg, config_folders)[1]
    attributions_shapley_baseline_convergence = load_attributions(cfg, config_folders, index_folder=1)[1]
    _, [convergence_level], _ = get_similarities(list_attributions=[attributions_shapley_baseline_convergence],
                                     references=attributions_shapley_baseline,
                                     which='pearson')

    cfgs_s_dependence = []
    for s in cfg.cardinalities:
        cfg_temp = deepcopy(cfg)
        cfg_temp.explainer.cardinality_coalitions = [s]
        cfgs_s_dependence.append(cfg_temp)

    list_attributions_varying_s = [load_attributions(cfg_temp, config_folders)[1] for cfg_temp in cfgs_s_dependence]
    _, similarity, similarity_error = get_similarities(list_attributions=list_attributions_varying_s, references=attributions_shapley_baseline,
                                     which='pearson')

    def get_ticks(cfg, type: str) -> [List[int], List[str]]:
        n_superpixel = cfg.explainer.segmentation.n_superpixel
        if type == 'dense':
            x_ticks_labels = [f'$\phi^{{n - {-s} }}$' if s < 0 else f'$\phi^{{{s}}}$'
                              for s in cfg.cardinalities]

            x_ticks_spacing = [- s if s < 0 else n_superpixel - s
                               for s in cfg.cardinalities]
        elif type == 'sparse':
            x_ticks_spacing = list(np.linspace(1, n_superpixel, 5, dtype=int))
            x_ticks_labels = [f'$\phi^{{n - {s} }}$' if s < int(n_superpixel / 2) else f'$\phi^{{{n_superpixel - s}}}$'
                              for s in x_ticks_spacing]
        elif type == 'percentage':
            x_ticks_spacing = list(np.linspace(1, n_superpixel, 5, dtype=int))
            x_ticks_labels = [f'{s / n_superpixel:.1f}'
                              for s in x_ticks_spacing]
        else:
            raise ValueError(f'type: {type} is not defined for spacing.')

        return x_ticks_spacing, x_ticks_labels

    type_ticks = 'percentage'
    x_ticks_spacing, x_ticks_labels = get_ticks(cfg, type=type_ticks)

    # def sparsify_ticks(spacing: List[int], labels: List[str]) -> [List[int], List[str]]:
    #     ticks_true = np.linspace(min(x_ticks_spacing), max(x_ticks_spacing), 5, dtype=int)
    #     index_ticks_true = [index for index, x_space in enumerate(spacing) if x_space == ]
    #     print(ticks_true)
    #     spacing_sparse = [spacing[i] for i in ticks_true]
    #     labels_sparse = [labels[i] for i in ticks_true]
    #     return spacing_sparse, labels_sparse
    #
    # x_ticks_spacing2, x_ticks_labels2 = sparsify_ticks(x_ticks_spacing, x_ticks_labels)


    x_ticks_spacing_all = [- s if s < 0 else cfg.explainer.segmentation.n_superpixel - s
                       for s in cfg.cardinalities]
    line2d = line_with_shaded_errorband(ax, x=np.array(x_ticks_spacing_all), y=similarity, yerr=similarity_error,
                                        label=map_imputer_name(cfg.explainer.imputer.name, what='name'),
                                        color=map_imputer_name(cfg.explainer.imputer.name, what='color'))
    # [line2d] = plt.plot(x_ticks_spacing_all, similarity, label=cfg.explainer.imputer.name)
    plt.axhline(convergence_level, linestyle='--', c=line2d.get_color())
    plt.xticks(ticks=x_ticks_spacing, labels=x_ticks_labels)
    if type_ticks == 'percentage':
        plt.xlabel(r'Occluded percentage $\frac{n-s}{n}$')
    plt.ylabel(r'$\langle \phi, \phi^s \rangle$')
    # plt.ylim((0.65, 1.005))
    plt.tight_layout(pad=0.1)
