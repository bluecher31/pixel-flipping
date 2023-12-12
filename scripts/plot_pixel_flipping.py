import copy

import warnings

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import seaborn as sns


import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, kendalltau, spearmanr
from sklearn.metrics import ndcg_score

from pathlib import Path
import itertools

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from src.plotting import summarize_pixel_flipping_results, map_imputer_name, add_text, \
    map_method_name, calculate_auc
from src.config_matplotlib import update_rcParams, line_with_shaded_errorband, clear_plot_figma

from src.experiments.pixel_flipping import load_pixel_flipping, root_pixel_flipping
from src.config.config import PixelFlippingConfig
from src.config.helpers import resolve_imagenet, load_all_configs, load_special_config
from src.interface import Measurement

from numpy.typing import NDArray
from typing import Dict, List


def get_results(cfg, config_folders) -> [Measurement, Measurement]:
    cfg_tmp = copy.deepcopy(cfg)
    cfg_tmp.most_relevant_first = True if cfg_tmp.pf_metric in ['MIF', 'NEW', 'MRG', 'SRG'] else False
    pf_results = load_pixel_flipping(cfg=cfg_tmp, config_folders=config_folders)
    x, y = summarize_pixel_flipping_results(pf_results)
    if cfg.pf_metric in ['NEW', 'SRG']:
        cfg_tmp.most_relevant_first = False
        pf_results_lif = load_pixel_flipping(cfg=cfg_tmp, config_folders=config_folders)
        x_lif, y_lif = summarize_pixel_flipping_results(pf_results_lif)
        x_new = x_lif - x
        assert np.alltrue(np.abs(x_new.values) <= 100 * x_new.error), \
            f'Occluded percentages do not match: \n{x.values} != {x_lif.values}'
        y = y_lif - y
    elif cfg.pf_metric == 'MRG' or cfg.pf_metric == 'LRG':
        # load random baseline
        x_random, y_random = average_random_result(cfg)

        x_new = x_random - x
        assert np.alltrue(np.abs(x_new.values) <= 100 * x_new.error), \
            f'Occluded percentages do not match: \n{x.values} != {x_random.values}'
        if cfg.pf_metric == 'MRG':
            y = y_random - y
        else:
            y = y - y_random
    return x, y


def average_random_result(cfg) -> [Measurement, Measurement]:
    cfg_tmp = copy.deepcopy(cfg)
    cfg_tmp.pf_metric = 'MIF'
    x_most, y_most = get_random_results(cfg_tmp)

    cfg_tmp.pf_metric = 'LIF'
    x_least, y_least = get_random_results(cfg_tmp)

    x_random = 1 / 2 * (x_most + x_least)
    y_random = 1 / 2 * (y_most + y_least)
    return x_random, y_random


def get_random_results(cfg) -> [Measurement, Measurement]:
    cfg_random = copy.deepcopy(cfg)
    cfg_random.explainer = None
    cfg_random.explainer = OmegaConf.create({"name": 'random'})
    root = root_pixel_flipping(dataset_name=cfg_random.dataset.name, model_name=cfg_random.model.name,
                               imputer_name_pf=cfg_random.imputer_pf.name, explainer_name=cfg_random.explainer.name)
    config_folders = load_all_configs(root=root)
    return get_results(cfg_random, config_folders)


def get_lrp_results(cfg) -> [Measurement, Measurement]:
    root = root_pixel_flipping(dataset_name=cfg.dataset.name, model_name=cfg.model.name,
                               imputer_name_pf=cfg.imputer_pf.name, explainer_name='zennit')
    config_folders = load_all_configs(root=root)
    cfg_lrp = copy.deepcopy(cfg)
    cfg_lrp.explainer = load_special_config(name='zennit', type='explainer')
    return get_results(cfg_lrp, config_folders)


def get_explainer_results(cfg, explainer_cfg_name: str) -> [Measurement, Measurement]:
    cfg_lrp = copy.deepcopy(cfg)
    cfg_lrp.explainer = load_special_config(name=explainer_cfg_name, type='explainer')
    root = root_pixel_flipping(dataset_name=cfg.dataset.name, model_name=cfg.model.name,
                               imputer_name_pf=cfg.imputer_pf.name, explainer_name=cfg_lrp.explainer.name)
    config_folders = load_all_configs(root=root)
    return get_results(cfg_lrp, config_folders)


def get_s_shapely_results(cfg, s: [int, None]) -> [Measurement, Measurement]:
    root = root_pixel_flipping(dataset_name=cfg.dataset.name, model_name=cfg.model.name,
                               imputer_name_pf=cfg.imputer_pf.name, explainer_name='Shapley values')
    config_folders = load_all_configs(root=root)
    cfg_tmp = copy.deepcopy(cfg)
    cfg_tmp.explainer.cardinality_coalitions = [s] if s is not None else None
    return get_results(cfg_tmp, config_folders)


def calculate_difference(m1: Measurement, m2: Measurement, operation: str) -> Measurement:
    if operation == '-':
        diff = m1.values - m2.values
    elif operation == 'mean':
        diff = (m1.values + m2.values) / 2
    else:
        raise NotImplementedError(f'This operation is not defined: {operation}')
    error = np.sqrt(m1.error**2 + m2.error**2)
    return Measurement(values=diff, error=error)


def line_plot(ax: plt.Axes, x_meas: Measurement, y_meas: Measurement, verbose_auc: bool = True, label: str = '',
              **kwargs) -> plt.matplotlib.lines.Line2D:
    line2d = line_with_shaded_errorband(ax, x=x_meas.values, y=y_meas.values, yerr=y_meas.error, label=label, **kwargs)
    ax.legend()
    ax.set_xlabel('Occlusion fraction')
    ax.set_ylabel('R-OMS')
    plt.tight_layout(pad=0.1)
    if verbose_auc:
        auc = calculate_auc(x_meas.values, y_meas.values, xmin=0, xmax=1)
        print(f'label = {label:<20} ->     auc = {auc:.3f}')

    return line2d


def add_auc(df: pd.DataFrame, x: [NDArray, Measurement], y: [NDArray, Measurement], method: str, info_dict: Dict,
            xmin: float = 0., xmax: float = 1.) -> pd.DataFrame:
    if isinstance(x, Measurement):
        x = x.values
    if isinstance(y, Measurement):
        y = y.values
    xmin = max(xmin, x.min())
    xmax = min(xmax, x.max())
    auc = calculate_auc(x, y, xmin=xmin, xmax=xmax)
    df_new = pd.DataFrame({'method': method, 'auc': auc, 'xmin': xmin, 'xmax': xmax,
                           'x_occluded_fraction': [x], 'f_remaining': [y], **info_dict},
                          index=[0])
    return pd.concat([df, df_new], ignore_index=True).sort_values(by='auc')


def compare_to_baseline(df: pd.DataFrame, key_reference: str, value_reference) -> pd.DataFrame:
    auc_reference = float(df.loc[df[key_reference] == value_reference]['auc'])

    df['auc_diff'] = df['auc'] - auc_reference
    df['auc_normalized'] = df['auc'] / auc_reference
    df['auc_diff_normalized'] = df['auc_diff'] / auc_reference
    df['auc_baseline'] = auc_reference
    df = df.sort_values(by='auc')
    return df


def s_shapley_auc_scores(cfg: PixelFlippingConfig):
    df = pd.DataFrame(columns=['method', 'auc', 'xmin', 'xmax'])
    list_xrange = [(0, 1), (0, 0.125), (0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1)]
    x_baseline, y_baseline = get_s_shapely_results(cfg, s=None)
    for xmin, xmax in list_xrange:
        df = add_auc(df, x_baseline, y_baseline, xmin=xmin, xmax=xmax, method='full_shapley', info_dict={})

    for i, s in enumerate(cfg.cardinalities):
        x_s, y_s = get_s_shapely_results(cfg, s=s)
        for xmin, xmax in list_xrange:
            df = add_auc(df, x_s, y_s, xmin=xmin, xmax=xmax, method=f's: {s}', info_dict={})

    df_grouped = df.groupby(by=['xmin', 'xmax'])
    # df.sort_values(by=['xmin', 'xmax'])
    for state, frame in df_grouped:
        print(f'xrange = {state}')
        frame_new = compare_to_baseline(frame, key_reference='method', value_reference='full_shapley')
        print(frame_new)


def s_shapley_pixel_flipping(cfg: PixelFlippingConfig):
    px = 2 / plt.rcParams['figure.dpi']  # pixel in inches
    if cfg.figma:
        figsize = (200*px, 90*px)
    else:
        figsize = (6, 3)
    fig = plt.figure(f'pf s-Shapley - {cfg.imputer_pf.name} - {cfg.model.name} - {cfg.segmentation_pf.n_superpixel}' + f'_{cfg.figma}',
                     figsize=figsize)
    ax = fig.add_subplot()

    # x_random, y_random = get_random_results(cfg)
    # line_plot(ax, x_random, y_random, label='random baseline')
    colors = plt.cm.viridis(np.linspace(0, 1, len(cfg.cardinalities)))

    x_baseline, y_baseline = get_s_shapely_results(cfg, s=None)
    if cfg.s_shapley_relative_to_baseline is False:
        line_plot(ax, x_baseline, y_baseline, label=f'full')

    for i, s in enumerate(cfg.cardinalities):
        x_s, y_s = get_s_shapely_results(cfg, s=s)
        # s_positive = s if s >= 0 else cfg.segmentation_pf.n_superpixel - abs(s)
        if np.allclose(x_baseline.values, x_s.values, atol=0.01) == False:
            raise RuntimeError('The x-values do not match. \nThe measurement have been taken at different occlusion points s.')
            # assert np.linalg.norm((x_baseline - x_s).values) < 0.001, \
            #     'The x-values do not match. \nThe measurement have been taken at different occlusion points s.'
        if cfg.s_shapley_relative_to_baseline:
            line_plot(ax, x_s, y_s - y_baseline, label=f's=[{s}]', color=colors[i])
        else:
            line_plot(ax, x_s, y_s, label=f's=[{s}]', color=colors[i])

    plt.axhline(0, label='full')

    ax.legend()
    # ax.set_title(f'imputer: {cfg.imputer_pf.name}')
    ax.set_xlabel('percentage occluded')
    if cfg.s_shapley_relative_to_baseline:
        ax.set_ylabel(r'pf[$\phi^s$] - pf[$\phi$]')
    plt.tight_layout(pad=0.1)

    if cfg.figma:
        clear_plot_figma()


def pf_pixelwise_explainers(cfg: PixelFlippingConfig) -> pd.DataFrame:
    df = pd.DataFrame()
    info_dict = {'imputer_pf': cfg.imputer_pf.name}

    title = f'pf - {cfg.imputer_pf.name} - {cfg.model.name} - {cfg.segmentation_pf.n_superpixel}'
    fig = plt.figure(title, figsize=(3, 2))
    ax = fig.add_subplot()
    # ax.set_title(title)

    n_repeat = 0
    for i, explainer_str in enumerate(cfg.explainers):
        x_occluded, f_remaining = get_explainer_results(cfg, explainer_str)
        df = add_auc(df, x_occluded, f_remaining, method=explainer_str, info_dict=info_dict)
        if '_nt' in explainer_str or '_abs' in explainer_str:
            linestyle = 'dashed'
            color = line2d.get_color()
            n_repeat += 1
        else:
            linestyle = 'solid'
            color = map_method_name(method=explainer_str, what='color')
            # color = f'C{i - n_repeat + 1}'
        line2d = line_plot(ax, x_occluded, f_remaining, label=map_method_name(explainer_str, what='name'),
                           linestyle=linestyle, color=color)
    # df = compare_to_baseline(df, key_reference='method', value_reference='random')
    return df


def auc_pixelwise_attributions(cfg):
    """Collect all experiments into a df."""
    models = ['resnet50', 'timm']
    n_superpixels = [25, 100, 500, 5000]
    list_results = []
    counter = 0
    for model in models:
        cfg.model = load_special_config(model, type='model')
        for n_superpixel in n_superpixels:
            cfg.segmentation_pf.n_superpixel = n_superpixel
            for imputer_name in cfg.imputers:
                cfg.imputer_pf = load_special_config(imputer_name, type='imputer')
                df = pf_pixelwise_explainers(cfg)
                plt.close(plt.gcf())
                df[['model', 'n_superpixel', 'identifier_exp']] = \
                    [cfg.model.name, cfg.segmentation_pf.n_superpixel, counter]
                list_results.append(df)
                counter += 1
    df_auc = pd.concat(list_results, ignore_index=True).sort_values(by='auc')
    folder_path = Path.cwd().parent / 'imagenet'
    assert folder_path.exists()
    df_auc.to_csv(folder_path / 'df_pixelwise_auc_test.csv')


def shapley_label(s) -> str:
    if s is None:
        return 'Shapley value'
    elif s == -1:
        return 'PredDiff'
    elif s == 0:
        return 'ArchAttribute'
    else:
        return s


def pf_overview(cfg: PixelFlippingConfig):
    markersize = 25
    if cfg.zoom_overview:
        figsize = (4, 3)
    else:
        figsize = (6, 3)
    fig = plt.figure(f'pf - {cfg.imputer_pf.name} - {cfg.model.name} - {cfg.segmentation_pf.n_superpixel} - {cfg.pf_metric}',
                     figsize=figsize)
    ax = fig.add_subplot()

    x_random, y_random = get_random_results(cfg)
    line_plot(ax, x_random, y_random, label='random baseline', color='grey')

    n_repeat = 0
    for i, explainer_str in enumerate(['ig_abs', 'gradients', 'input_x_gradients', 'zennit', 'ig', 'ig_nt', 'gradients_nt']):
        x_occluded, f_remaining = get_explainer_results(cfg, explainer_str)
        if '_nt' in explainer_str:
            linestyle = 'dashdot'
            color = line2d.get_color()
            n_repeat += 1
        else:
            linestyle = 'dashed'
            color = f'C{i - n_repeat}'
        line2d = line_plot(ax, x_occluded, f_remaining, label=explainer_str, linestyle=linestyle, color=color,
                           marker='X', markersize=np.sqrt(markersize))

    imputer_overview = cfg.imputer_overview if cfg.imputer_pf.name == 'diffusion' else cfg.imputer_pf.name
    imputer_cfg = load_special_config(imputer_overview, type='imputer')
    cfg.explainer.imputer = imputer_cfg
    for j, s in enumerate(['None', -1, 0]):
        if s == 'None':
            s = None
        x, y = get_s_shapely_results(cfg, s=s)
        linestyle = 'solid'
        color = f'C{i + j - n_repeat}'
        line2d = line_plot(ax, x, y, label=shapley_label(s), linestyle=linestyle, color=color,
                           marker='D', markersize=np.sqrt(markersize))

    if cfg.zoom_overview:
        ax.set_xlim(-0.005, 0.13)
        ax.set_xticks(ticks=[0, 0.05, 0.1])
        ax.set_yticks(ticks=[0.5, 0.6, 0.7])
        ax.set_ylim(0.46, 0.71)
        ax.yaxis.set_ticks_position('right')
        ax.xaxis.set_ticks_position('top')

        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')
        ax.set_xlabel('')
        ax.set_ylabel('')
        plt.tight_layout(pad=0.1)
        ax.get_legend().remove()


def pf_imputers(cfg: PixelFlippingConfig):
    fig = plt.figure(f'pf - {cfg.imputer_pf.name} - {cfg.model.name} - {cfg.segmentation_pf.n_superpixel}')
    ax = fig.add_subplot()

    x_random, y_random = get_random_results(cfg)
    line_plot(ax, x_random, y_random, label='random baseline')

    root = root_pixel_flipping(dataset_name=cfg.dataset.name, model_name=cfg.model.name,
                               imputer_name_pf=cfg.imputer_pf.name, explainer_name=cfg.explainer.name)
    config_folders = load_all_configs(root=root)
    for imputer_name in cfg.imputers:
        imputer_cfg = load_special_config(imputer_name, type='imputer')
        cfg.explainer.imputer = imputer_cfg
        x, y = get_results(cfg, config_folders)
        line_plot(ax, x, y, label=f'{cfg.explainer.name:.7} - {cfg.explainer.imputer.name:.10}')


def imputer_dependence_auc_scores(cfg: PixelFlippingConfig):
    df = pd.DataFrame()
    for imputer_name_pf in cfg.imputers:
        imputer_cfg = load_special_config(imputer_name_pf, 'imputer')
        cfg.imputer_pf = imputer_cfg

        root = root_pixel_flipping(dataset_name=cfg.dataset.name, model_name=cfg.model.name,
                                   imputer_name_pf=cfg.imputer_pf.name, explainer_name=cfg.explainer.name)
        config_folders = load_all_configs(root=root)
        for imputer_name_explainer in cfg.imputers:
            if imputer_name_explainer == 'diffusion':
                continue
            imputer_cfg = load_special_config(imputer_name_explainer, 'imputer')
            cfg.explainer.imputer = imputer_cfg
            x, y = get_results(cfg, config_folders)
            df = add_auc(df, x, y, method='full_shapley',
                         info_dict={'imputer_explainer': cfg.explainer.imputer.name, 'imputer_pf': cfg.imputer_pf.name})

    print(df)
    df_grouped = df.groupby(by=['imputer_pf'])
    # df.sort_values(by=['xmin', 'xmax'])
    list_res = []
    for state, frame in df_grouped:
        # print(f'imputer_pf = {state}')
        frame_new = compare_to_baseline(df=frame, key_reference='imputer_explainer', value_reference=state)
        print(frame_new)
        list_res.append({state: frame_new[['imputer_explainer', 'auc_normalized', 'auc_diff']]})

    for res in list_res:
        print(res)
    # print(list_res)


def occlusion_dependence_pf_metrics(cfg):
    explainer_str = cfg.explainers[0]

    title = f'pf - {explainer_str} - metric={cfg.pf_metric}'
    fig = plt.figure(title)
    ax = fig.add_subplot()
    ax.set_title(title)

    for imputer_name in cfg.imputers:
        imputer_cfg = load_special_config(imputer_name, type='imputer')
        cfg.imputer_pf = imputer_cfg
        x, y = get_explainer_results(cfg, explainer_str)
        line_plot(ax, x, y, label=f'{cfg.imputer_pf.name:.10}')


def plot_linear_trend(ax: plt.Axes, x: pd.Series, y: pd.Series):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # calculate equation for trendline
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)

    x_trend = np.linspace(xmin, xmax, 10)
    # add trendline to plot
    ax.plot(x_trend, p(x_trend), 'grey')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # x_optimal = np.linspace(xmin, xmax)


def scatter_experiments(ax: plt.Axes, x: pd.Series, y: pd.Series, color_labels: List, marker_labels: List, normalize: bool = False,
                        **kwargs):
    markers = ['o', 'D', 'X', 'v', '*', '>', '^', '<']
    markersize = 25
    if normalize:
        x /= x.max()
        y /= y.max()
    unique_color_labels = sorted(np.unique(color_labels))
    if len(unique_color_labels) == 5:            # manually sort imputers
        unique_color_labels = [unique_color_labels[i] for i in [1, 2, 0, 3, 4]]

    unique_marker_labels = np.unique(marker_labels)
    for i_color, c_label in enumerate(unique_color_labels):
        mask_color = np.array(color_labels) == c_label
        for i_marker, m_label in enumerate(unique_marker_labels):
            mask_marker = np.array(marker_labels) == m_label
            mask = np.logical_and(mask_color, mask_marker)
            ax.scatter(x[mask], y[mask], color=map_imputer_name(c_label, what='color'), marker=markers[i_marker], s=markersize,
                       **kwargs)

    # legend
    # label_column = ['A', 'B', 'C']
    # label_row = ['1', '2', '3']
    handle_color = [mpatches.Patch(color=map_imputer_name(c_label, what='color'))
                    for c_label in unique_color_labels]
    handle_marker = [plt.plot([], [], markers[i], markerfacecolor='k', markeredgecolor='k',
                              markersize=np.sqrt(markersize), alpha=0.6)[0]
                     for i in range(len(unique_marker_labels))]

    legend_marker = [int(n) for n in unique_marker_labels]
    legend_colors = [map_imputer_name(name, what='name') for name in unique_color_labels]
    ax.legend(handle_color + handle_marker, legend_colors + legend_marker, ncol=2,
              title='Imputer | $n$', title_fontsize='small')

    plt.tight_layout(pad=0.1)
    correlation = np.corrcoef(x, y)[0, 1]

    add_text(text=f'Pearson correlation: {correlation:.3f}', ax=ax, loc='lower right')


def set_xlim_r_oms(ax):
    ax.set_xlim(0.03, 0.592)


def average_distance_to_baseline(df: pd.DataFrame, title: str = ''):
    df_experiments = df.groupby(by='identifier_exp')
    df_analysis = pd.DataFrame()
    for i_exp, frame_exp in df_experiments:
        auc_random = float(np.unique(frame_exp['auc_baseline']))
        auc_diff_mean = frame_exp["auc_diff"].mean()
        auc_diff_var = frame_exp["auc_diff"].var()
        imputer_pf = np.unique(frame_exp['imputer_pf']).reshape(1)[0]
        n_superpixel = float(np.unique(frame_exp['n_superpixel']))

        mask_random = frame_exp['method'] != 'random'
        X = frame_exp[mask_random]['auc'].to_numpy()[:, None]
        pairwise_distances = pdist(X=X, metric='Euclidean')

        X = frame_exp[mask_random]['auc_normalized'].to_numpy()[:, None]
        relative_pairwise_distances = pdist(X=X, metric='Euclidean')

        df_new = pd.DataFrame({'auc_random': auc_random, 'auc_diff_var': auc_diff_var, 'auc_diff_mean': auc_diff_mean,
                               'imputer_pf': imputer_pf, 'n_superpixel': n_superpixel,
                               'mean_pairwise_distance': pairwise_distances.mean(),
                               'mean_relative_pairwise_distance': relative_pairwise_distances.mean()}, index=[0])
        df_analysis = pd.concat([df_analysis, df_new], ignore_index=True).sort_values(by='auc_random')
        # print(f'auc random: {auc_random:.4f}   -    variance: {df["auc_normalized"].var():.4f}')

    # plot_normalized(x=df_analysis['auc_random'], y=df_analysis['auc_diff_var'], label='var')
    title = 'pairwise_distance_between_methods' + title
    fig = plt.figure(title, figsize=(6.3, 2))
    ax = fig.add_subplot()

    # plt.plot(x, x, color='Grey', label='Perfect correlation')
    y = df_analysis['mean_pairwise_distance'] / df_analysis['mean_pairwise_distance'].max()
    scatter_experiments(ax=ax, x=df_analysis['auc_random'], y=y,
                        color_labels=df_analysis['imputer_pf'], marker_labels=df_analysis['n_superpixel'])
    plot_linear_trend(ax, df_analysis['auc_random'], y)
    set_xlim_r_oms(ax)
    plt.xlabel(r'AUC[R-OMS]')
    # plt.ylabel(r'$\langle \mathrm{AUC}[random] - \mathrm{AUC}[method] \rangle$')
    # plt.ylabel('Discriminability')
    plt.ylabel('Separation')
    # plt.ylim(-0.05, 1.05)
    # plt.xlim(-0.05, 1.05)
    plt.tight_layout(pad=0.1)

    print('Correlation between different measures: ')
    print(df_analysis[['auc_random', 'auc_diff_mean', 'auc_diff_var']].corr())


def imshow_ranking(df: pd.DataFrame, ranking_key='ranking_most',
                   sorting_variable: str = 'auc_random', title: str = ''):
        base_ranking = df[ranking_key].iloc[-1]
        n_methods = len(base_ranking)

        def map_to_int(method: str, base_ranking: List[str]) -> int:
            return base_ranking.index(method)

        if sorting_variable == 'random':
            from sklearn.utils import shuffle
            df_sorted = shuffle(df)
        else:
            ascending = False if sorting_variable in ['n_superpixel', 'imputer order 118', 'model order 0',
                                                      'distance-most_frequent',
                                                      f'distance_kendall_tau_{ranking_key}_numeric'] \
                else True
            df_sorted = df.sort_values(by=sorting_variable, ascending=ascending)

        rankings = []
        for ranking_str in df_sorted[ranking_key]:
            ranking_int = np.array([map_to_int(method=method, base_ranking=base_ranking)
                                    for method in ranking_str])
            rankings.append(ranking_int)
        ranking_np = np.stack(rankings)
        assert ranking_np.ndim == 2, 'Something went wrong. Except an two-dimensional array,'

        fig = plt.figure('ranking_imshow_' + sorting_variable + title, figsize=(6.3, 1.6))
        ax = fig.add_subplot()
        colors = [map_method_name(method, what='color') for method in base_ranking]
        cmap = ListedColormap(colors)
        if ranking_key in ['ranking_least', 'ranking_most_minus_least']:
            ranking_np = np.flip(ranking_np, axis=1)
        #
        # else:
        #     print('Warning: y-axis ticks are incorrect/flipped.')

        im = ax.imshow(ranking_np.T, cmap=cmap)
        plt.ylabel('Ranking method')
        plt.xlabel('Experiments (sorted by AUC[R-OMS])')
        yticks = np.arange(0, n_methods, int(n_methods/3))
        plt.yticks(ticks=yticks, labels=yticks + 1)

        xticks = np.linspace(0, len(ranking_np)-1, 3)
        plt.xticks(xticks, ['Low', 'Median', 'High'])

        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax, ticks=np.linspace(0.5, n_methods - 1.5, n_methods))
        cbar.ax.set_yticklabels(base_ranking)
        # # cbar.ax.axis["bottom"].major_ticklabels.set_va("top")

        plt.tight_layout(pad=0.1)
        plt.sca(ax)


def imshow_ranking_difference_to_MFR(df: pd.DataFrame, ranking_key='ranking_most',
                   title: str = ''):
    """MRF: most frequent ranking."""
    sorting_variable: str = 'auc_random'
    ranking_mfr = ['random', 'input_x_gradients', 'ig_nt', 'ig', 'gradients', 'gradients_nt', 'zennit']
    ranking_mfr_dict = {method: rank for rank, method in enumerate(ranking_mfr)}
    n_methods = len(ranking_mfr)

    def map_to_int(method: str, base_ranking: List[str]) -> int:
        return base_ranking.index(method)

    df_sorted = df.sort_values(by=sorting_variable, ascending=True)

    agreement_to_mfr = []
    for ranking_str in df_sorted[ranking_key]:
        # ranking_int = np.array([map_to_int(method=method, base_ranking=ranking_mfr)
        #                         for method in ranking_str])
        # rankings.append(ranking_int)
        ranking_displacement = []
        if ranking_key in ['ranking_most']:
            ranking_str = ranking_str[::-1]
        for rank_method, method in enumerate(ranking_str):
            target_rank = ranking_mfr_dict[method]
            abs_rank_displacement = np.abs(target_rank - rank_method)
            ranking_displacement.append(int(abs_rank_displacement))

        agreement_to_mfr.append(ranking_displacement)
    ranking_agreement_to_mfr_np = np.stack(agreement_to_mfr)
    # if ranking_key in ['ranking_least', 'ranking_most_minus_least']:
    ranking_agreement_to_mfr_np = np.flip(ranking_agreement_to_mfr_np, axis=1)
    assert ranking_agreement_to_mfr_np.ndim == 2, 'Something went wrong. Except an two-dimensional array,'

    fig = plt.figure('ranking_agreement_mfr' + sorting_variable + title, figsize=(6.3, 1.6))
    ax = fig.add_subplot()
    # colors = [map_method_name(method, what='color') for method in ranking_mfr]
    # cmap = ListedColormap(colors)

    im = ax.imshow( ranking_agreement_to_mfr_np.T, cmap='plasma', vmin=0, vmax=len(ranking_mfr))
    plt.ylabel('Ranking method')
    plt.xlabel('Experiments (sorted by AUC[R-OMS])')
    yticks = np.arange(0, n_methods, int(n_methods / 3))
    plt.yticks(ticks=yticks, labels=yticks + 1)

    xticks = np.linspace(0, len( ranking_agreement_to_mfr_np) - 1, 3)
    plt.xticks(xticks, ['Low', 'Median', 'High'])

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax, ticks=np.linspace(0.5, n_methods - 1.5, n_methods))
    # cbar.ax.set_yticklabels(ranking_mfr)

    plt.tight_layout(pad=0.1)
    plt.sca(ax)


def scatter_ranking_ig(df: pd.DataFrame):
    fig = plt.figure('ranking_ig', figsize=(6.3, 2))
    ax = fig.add_subplot()
    scatter_experiments(ax, x=df['auc_random'], y=df['rank_ig'], normalize=True,
                        color_labels=df['imputer_pf'], marker_labels=df['n_superpixel'])
    # plt.scatter(df_analysis['auc_random'], df_analysis['rank_ig'], label='Experiments')
    plt.xlabel('AUC[R-OMS]')
    plt.ylabel('rank IG')
    # plt.legend()
    plt.tight_layout(pad=0.1)


def scatter_alignment_most_vs_least(df: pd.DataFrame):
    fig = plt.figure('alignment_most_vs_least', figsize=(6.3, 2))
    ax = fig.add_subplot()

    x = 'auc_random'
    y = 'diff(most,least)'
    df_ = df[~df[y].isna()]
    scatter_experiments(ax, x=df_[x], y=df_[y], normalize=False,
                        color_labels=df_['imputer_pf'], marker_labels=df_['n_superpixel'])

    set_xlim_r_oms(ax)
    ax.set_xlabel('AUC[R-OMS]')
    ax.set_ylabel('Most vs. least')
    plt.tight_layout(pad=0.1)


def most_frequent(df_rankings):
    rankings, counts = np.unique(df_rankings.values, return_counts=True)
    most_frequent_ranking = rankings[counts.argmax()]
    return most_frequent_ranking


def ranking_distance_to_start(df_rankings, start_ranking, mode='correlation'):
    if mode == 'correlation':
        return df_rankings.apply(lambda r: pearsonr(start_ranking, r)[0])
    elif mode == 'kendall_tau':
        return df_rankings.apply(lambda r: kendall_tau_distance(start_ranking, r))
    elif mode == 'nDCG':
        def map_ranking_to_scores(ranking: List[int]) -> NDArray:
            scores = []
            for label_method in ranking:
                score = len(start_ranking) - np.argmax(np.array(start_ranking) == label_method)
                scores.append(score)
            return np.array([scores])

        y_true = map_ranking_to_scores(start_ranking)
        df_rankings.apply(lambda r: ndcg_score(y_true=np.array([start_ranking]), y_score=np.array([r])))
        return df_rankings.apply(lambda r: ndcg_score(y_true=y_true, y_score=map_ranking_to_scores(r)))
    else:
        raise ValueError(f'Not defined. mode = {mode}')


def table_sorting_observables(df: pd.DataFrame, ranking_type: str = 'ranking_most_numeric', mode='distance'):
    # merge maxf
    # df_final = df_rankings.merge(df_observables_maxf.set_index(['model', 'n_superpixel', 'imputer_pf'])[['auc_maxf']],
    #                              left_index=True, right_index=True, how='outer')
    # merge orgf
    # df_final = df_final.merge(df_observables_orgf.set_index(['model', 'n_superpixel', 'imputer_pf'])[['auc_orgf']],
    #                           left_index=True, right_index=True)

    df_final = df.copy()

    key_distance = f'distance_{mode}_{ranking_type}'

    most_frequent_ranking = most_frequent(df_final[ranking_type])
    df_final[f'distance_kendall_tau_{ranking_type}'] = ranking_distance_to_start(df_final[ranking_type], most_frequent_ranking,
                                                           mode='kendall_tau')

    df_final[f'distance_nDCG_{ranking_type}'] = ranking_distance_to_start(df_final[ranking_type], most_frequent_ranking,
                                                           mode='nDCG')

    # imputer order
    imputers = df_final['imputer_pf'].unique()
    imputer_order = np.array(list(itertools.permutations(np.arange(1, len(imputers) + 1))))
    imputer_order_columns = [f'imputer order {i}' for i in range(imputer_order.shape[0])]
    imputer_df = pd.DataFrame(imputer_order.T, index=imputers,
                              columns=imputer_order_columns)
    df_final = df_final.merge(imputer_df, left_on='imputer_pf', right_index=True)

    # model order
    models = df_final['model'].unique()
    model_order = np.array(list(itertools.permutations(np.arange(1, len(models) + 1))))
    model_df = pd.DataFrame(model_order.T, index=models,
                            columns=[f'model order {i}' for i in range(model_order.shape[0])])
    df_final = df_final.merge(model_df, left_on='model', right_index=True)

    df_final['random_sort'] = np.random.choice(np.arange(len(df_final)), replace=False, size=len(df_final))

    df_corr = df_final.corr(method='spearman').round(2)

    # imputer_order_max = df_corr.loc['auc_orgf'][imputer_df.columns].idxmax()
    # to_drop = imputer_df.columns.drop(imputer_order_max)
    # df_corr = df_corr.drop(to_drop, axis=0).drop(to_drop, axis=1)

    # get imputer order which maximizes correlation
    df_corr_imputer = df_corr.loc[imputer_order_columns][[key_distance]]
    imputer_order_max_corr = df_corr_imputer.idxmax()[key_distance]
    # df_corr_imputer.loc[imputer_order_max_corr['distance-most_frequent']]
    print('---')
    rankings_np = np.stack(df_final[ranking_type])
    n_unique_rankings = len(np.unique(rankings_np, axis=0))
    print(f'{ranking_type.upper()}: number of unique rankings = {n_unique_rankings}')
    print(f'most frequent ranking:\n{most_frequent_ranking}')
    print(df_corr.loc[['auc_random', 'max_f', 'n_superpixel', imputer_order_max_corr, 'model order 0', 'random_sort']][[key_distance]])
    df_final_filtered = df_final[df.keys().append(pd.Index(['model order 0', imputer_order_max_corr,
                                                            f'distance_kendall_tau_{ranking_type}',
                                                            f'distance_nDCG_{ranking_type}']
                                                           ))]
    return df_final_filtered


def kendall_tau_distance(order_a: List[int], order_b: List[int], normalize=False):
    if (isinstance(order_a, list) and isinstance(order_b, list)) is False:
        return np.NAN
    # try:
    #     len(order_a) == len(order_b)
    # except TypeError:
    #     print('error')
    assert len(order_a) == len(order_b), f'Different number of items in orderings a,b.\n {order_a}\n{order_b}'
    pairs = itertools.combinations(range(1, len(order_a)+1), 2)
    distance = 0
    for x, y in pairs:
        a = order_a.index(x) - order_a.index(y)
        b = order_b.index(x) - order_b.index(y)
        if a * b < 0:
            distance += 1
    if normalize:
        n = len(order_a)
        distance = distance/(0.5*n*(n-1))

    return distance


def extract_rankings(df: pd.DataFrame) -> pd.DataFrame:
    df_experiments = df.groupby(by='identifier_exp')
    df_analysis = pd.DataFrame()
    # rankings = []
    for i_exp, frame_exp in df_experiments:
        frame = frame_exp.sort_values(by='auc')
        auc_random = float(np.unique(frame_exp['auc_baseline']))
        ranking = list(frame['method'])
        imputer_pf = np.unique(frame_exp['imputer_pf']).reshape(1)[0]
        n_superpixel = float(np.unique(frame_exp['n_superpixel']))
        model = np.unique(frame_exp['model'])[0]
        if 'ig' not in ranking:
            continue
        df_new = pd.DataFrame({'auc_random': auc_random, 'ranking': [ranking], 'rank_ig': ranking.index('ig'),
                               'imputer_pf': imputer_pf, 'n_superpixel': n_superpixel, 'model': model}, index=[0])
        df_analysis = pd.concat([df_analysis, df_new], ignore_index=True).sort_values(by='auc_random')
    return df_analysis


def create_analyze_rankings_df(df_most: pd.DataFrame, df_least: pd.DataFrame, df_f_max: pd.DataFrame) \
        -> [pd.DataFrame, pd.DataFrame]:
    methods = sorted(df_most['method'].unique())
    assert methods == sorted(df_least['method'].unique()), f'Check for matching methods.\n {methods}'

    df_ranking_most = extract_rankings(df_most)
    df_ranking_least = extract_rankings(df_least)

    df_most_merge = df_most[['method', 'auc', 'n_superpixel', 'imputer_pf', 'model', 'identifier_exp', 'auc_baseline']]
    df_most_merge.rename({'auc': 'auc_most'}, axis='columns', inplace=True)
    df_least_merge = df_least[['method', 'auc', 'n_superpixel', 'imputer_pf', 'model', 'identifier_exp']]
    df_least_merge.rename({'auc': 'auc_least'}, axis='columns', inplace=True)

    df_most_least_auc = pd.merge(df_most_merge, df_least_merge, how='left')
    df_most_least_auc['auc'] = df_most_least_auc['auc_least'] - df_most_least_auc['auc_most']
    df_ranking_least_minus_most = extract_rankings(df_most_least_auc)
    df_ranking_least_minus_most.rename({'ranking': 'ranking_most_minus_least'}, axis='columns', inplace=True)

    # add ranking least relevant first
    df_ranking_least_merge = df_ranking_least[['n_superpixel', 'imputer_pf', 'model',  'ranking']]
    df_ranking_least_merge.rename({'ranking': 'ranking_least'}, axis='columns', inplace=True)
    df_ranking_most.rename({'ranking': 'ranking_most'}, axis='columns', inplace=True)
    df_rankings = pd.merge(df_ranking_most, df_ranking_least_merge, how='left')

    df_max_merge = df_f_max[['n_superpixel', 'imputer_pf', 'model',  'auc']]
    df_max_merge.rename({'auc': 'max_f'}, axis='columns', inplace=True)
    df_rankings = pd.merge(df_rankings, df_max_merge, how='left')
    # add NR-OMS

    df_rankings = pd.merge(df_rankings, df_ranking_least_minus_most[['n_superpixel', 'imputer_pf', 'model', 'ranking_most_minus_least']], how='left')

    # diff(most, least)
    method_id_map = {m: i + 1 for i, m in enumerate(methods)}

    def map_ranking_to_numeric(list_ranking: List[str]) -> List[int]:
        if isinstance(list_ranking, list) is False:
            return np.NAN
        return [method_id_map[method] for method in list_ranking]

    df_rankings['ranking_most_numeric'] = df_rankings['ranking_most'].apply(map_ranking_to_numeric)
    df_rankings['ranking_least_numeric'] = df_rankings['ranking_least'].apply(map_ranking_to_numeric)
    df_rankings['ranking_NEW_numeric'] = df_rankings['ranking_most_minus_least'].apply(map_ranking_to_numeric)

    df_rankings['diff(most,least)'] = df_rankings.apply(
        lambda x: kendall_tau_distance(x['ranking_most_numeric'], x['ranking_least_numeric'], normalize=True), axis=1)

    methods_numeric = map_ranking_to_numeric(methods)
    random_rankings = pd.Series([list(np.random.choice(methods_numeric, len(methods_numeric), replace=False))
                                 for _ in range(len(df_rankings))])
    df_rankings['ranking_random'] = random_rankings
    df_rankings['diff(most,random)'] = df_rankings.apply(
        lambda x: kendall_tau_distance(x['ranking_most_numeric'], x['ranking_random'], normalize=True), axis=1)
    df_rankings['diff(least,random)'] = df_rankings.apply(
        lambda x: kendall_tau_distance(x['ranking_least_numeric'], x['ranking_random'], normalize=True), axis=1)

    df_rankings.sort_values(by='auc_random', inplace=True)
    return df_rankings, df_most_least_auc


def select_experiments(df: pd.DataFrame, imputer: str = None, n_superpixel: int = None, model: str = None) -> pd.DataFrame:
    """Filter for all provided arguments."""
    df_tmp1 = df[df['imputer_pf'] == imputer] if imputer is not None else df
    df_tmp2 = df_tmp1[df_tmp1['model'] == model] if model is not None else df
    df_final = df_tmp2[df_tmp2['n_superpixel'] == n_superpixel] if n_superpixel is not None else df
    return df_final


def contradiction_intro_table(df: pd.DataFrame):
    df_exp1 = select_experiments(df=df, imputer='TrainSet', n_superpixel=25, model='ResNet50')
    df_exp2 = select_experiments(df=df, imputer='TrainSet', n_superpixel=500, model='ResNet50')
    df_exp3 = select_experiments(df=df, imputer='diffusion', n_superpixel=500, model='ResNet50')
    for df_exp in [df_exp1, df_exp2, df_exp3]:
        assert len(df_exp) == 1, 'Expect a single PF setup.'
        dict_exp = df_exp.iloc[0].to_dict()
        print('---')
        print(f'setup: model={dict_exp["model"]:s} - imputer={dict_exp["imputer_pf"]} - n_superpixel={dict_exp["n_superpixel"]:.0f}')
        print(f'MIF: {dict_exp["ranking_most"]}')
        print(f'LIF: {dict_exp["ranking_least"][::-1]}')
        print(f'NEW: {dict_exp["ranking_most_minus_least"][::-1]}')
        print('---')


def correlation_to_random_ranking(df: pd.DataFrame):
    key_distance = 'distance_nDCG_ranking_most_numeric' if 'distance_nDCG_ranking_most_numeric' in df.columns else \
        'distance_nDCG_ranking_least_numeric'
    random_corrs = []
    for _ in range(100):
        df['random_sort'] = np.random.choice(np.arange(len(df)), replace=False, size=len(df))
        df_corr = df.corr(method='spearman').round(5)
        random_corr = float(df_corr.loc[['random_sort']][[key_distance]].iloc[0, 0])
        random_corrs.append(random_corr)
    print(f'{key_distance}: random = {np.mean(random_corrs):.4f} +- {np.std(random_corrs):.4f}')


def global_beeplot_xai_methods(df: pd.DataFrame):
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(6, 3.3))
    beeplot_xai_methods(ax=axs[0], df=df, key_auc='auc_mif_tmp', title='Most relevant', legend=True)
    axs[0].set_xlabel('MRG')

    beeplot_xai_methods(ax=axs[1], df=df, title='Two-sided PF', yticks=False, legend=False)
    axs[1].set_xlabel('SRG')

    beeplot_xai_methods(ax=axs[2], df=df, key_auc='auc_lif_tmp', title='Least relevant', yticks=False, legend=False)
    axs[2].set_xlabel('LRG')
    plt.tight_layout(pad=0.1)


def beeplot_xai_methods(ax: plt.Axes, df: pd.DataFrame, key_auc: str = 'auc', title: str = '', yticks: bool = True,
                        legend: bool = True):
    imputers = ['ConstantValueImputer', 'TrainSet', 'ColorHistogram', 'cv2', 'diffusion']
    palette = {imputer: map_imputer_name(imputer, what='color') for imputer in imputers}
    methods = ['zennit', 'gradients_nt', 'gradients', 'ig', 'ig_nt', 'input_x_gradients']

    # sns.catplot(data=df, x=key_auc, y="method", hue="imputer_pf", kind="box", legend=False, palette=palette,
    #             order=methods, hue_order=imputers[::-1], height=3.5, aspect=3./3.5)

    sns.boxplot(ax=ax, data=df, x=key_auc, y="method", hue="imputer_pf", palette=palette,
                order=methods, hue_order=imputers[::-1])

    ax.set_title(title, fontdict={'size': 12})
    ax.set_ylabel('')
    ax.set_xlabel(r'$\Delta$-AUC over random baseline')
    if yticks:
        labels = [map_method_name(m, what='name') if m != 'input_x_gradients' else 'InputX\nGradients'
                  for m in methods ]
        ax.set_yticks(labels=labels, ticks=ax.get_yticks(), rotation=30, va='center', ha='right')
    else:
        ax.set_yticks(labels=['' for _ in methods], ticks=ax.get_yticks())
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        imputer_labels = [map_imputer_name(imputer, what='name') for imputer in labels]
        ax.legend(handles, imputer_labels, loc='lower right')
    else:
        ax.get_legend().remove()
    plt.tight_layout(pad=0.1)


cs = ConfigStore.instance()
cs.store(name="config", node=PixelFlippingConfig)
OmegaConf.register_new_resolver('resolve_imagenet', resolve_imagenet)


@hydra.main(config_path='../src/conf', config_name='plot_pixel_flipping', version_base=None)
def main(cfg: PixelFlippingConfig):
    print(OmegaConf.to_yaml(cfg))

    if 'pf_imputers' in cfg.plotting_routines:
        pf_imputers(cfg)
    if 'pf_overview' in cfg.plotting_routines:
        for imputer_name in cfg.imputers:
            cfg.imputer_pf = load_special_config(imputer_name, 'imputer')
            pf_overview(cfg)
    if 'pf_pixelwise_attributions' in cfg.plotting_routines:
        list_results = []
        for imputer_name in cfg.imputers:
            cfg.imputer_pf = load_special_config(imputer_name, 'imputer')
            df = pf_pixelwise_explainers(cfg)
            list_results.append(df)
    if 'auc_pixelwise_experiments' in cfg.plotting_routines:
        auc_pixelwise_attributions(cfg)
    if 's_shapley' in cfg.plotting_routines:
        s_shapley_pixel_flipping(cfg)
    if 's_shapley_auc' in cfg.plotting_routines:
        s_shapley_auc_scores(cfg)
    if 'imputer_dependence_auc' in cfg.plotting_routines:
        imputer_dependence_auc_scores(cfg)
    if 'occlusion_dependence_pf_metrics' in cfg.plotting_routines:
        occlusion_dependence_pf_metrics(cfg)
    if 'analyze_rankings' in cfg.plotting_routines:
        df_most = pd.read_csv(Path.cwd().parent / '../data/df_pixelwise_auc.csv')
        df_least = pd.read_csv(Path.cwd().parent / '../data/df_pixelwise_least_auc.csv')
        df_maxf = pd.read_csv(Path.cwd().parent / '../data/df_pf_max_f.csv')

        df_rankings, df_new_measure = create_analyze_rankings_df(df_most=df_most, df_least=df_least, df_f_max=df_maxf)
        df_new_measure['auc_mif_tmp'] = - (df_most['auc_diff'])
        df_new_measure['auc_lif_tmp'] = df_least['auc'] - df_least['auc_baseline']


        for ranking_key in ['ranking_most', 'ranking_least', 'ranking_most_minus_least']:
            pass
            imshow_ranking(df=df_rankings, ranking_key=ranking_key, title=ranking_key)
            imshow_ranking_difference_to_MFR(df=df_rankings, ranking_key=ranking_key, title=ranking_key)

        # mask_lrp_wins = np.stack([row[0] == 'zennit' for row in df_rankings['ranking_most']])
        # mask_standard_resnet = np.stack([row == 'ResNet50' for row in df_rankings['model']])

        # average_distance_to_baseline(df_most, title='MIF')
        # average_distance_to_baseline(df_least, title='LIF')

        # beeplot_xai_methods(df_most, title='MIF')
        # beeplot_xai_methods(df_least, title='LIF')
        global_beeplot_xai_methods(df_new_measure)

        contradiction_intro_table(df_rankings)


        # plt.title('Area between least vs. most relevant PF curves')
        mode = 'nDCG'       # distance or nDCG
        df_design_variables_most = table_sorting_observables(df=df_rankings, mode=mode)
        df_design_variables_least = table_sorting_observables(df=df_rankings, ranking_type='ranking_least_numeric',
                                                              mode=mode)
        df_design_variables_new = table_sorting_observables(df=df_rankings, ranking_type='ranking_NEW_numeric',
                                                            mode=mode)

        correlation_to_random_ranking(df_design_variables_most)
        correlation_to_random_ranking(df_design_variables_least)

        for ranking_key in ['ranking_most', 'ranking_least']:
            list_sorting_variable = []
            # list_sorting_variable += [f'distance_nDCG_{ranking_key}_numeric']
            # list_sorting_variable += [f'distance_kendall_tau_{ranking_key}_numeric']
            list_sorting_variable += ['auc_random']
            # list_sorting_variable += ['auc_random', 'random', 'n_superpixel']
            # list_sorting_variable.append('imputer order 118' if ranking_key == 'ranking_most' else 'imputer order 80')
            # list_sorting_variable += ['max_f', 'model order 0']

            # for sorting_variable in list_sorting_variable:
            #     df_design_variables = df_design_variables_most if ranking_key == 'ranking_most' \
            #         else df_design_variables_least
            #     imshow_ranking(df=df_design_variables, sorting_variable=sorting_variable, ranking_key=ranking_key,
            #                    title=f' - {ranking_key}')
            #     plt.title(sorting_variable)

        # scatter_ranking_ig(df=df_rankings)

        # scatter_alignment_most_vs_least(df=df_rankings)


if __name__ == '__main__':
    warnings.simplefilter("ignore")
    update_rcParams(fig_width_pt=234.88 * 0.85, half_size_image=True)

    main()

