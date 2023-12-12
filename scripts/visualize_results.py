import torch
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from src.plotting import visualize_attribution, map_attribution_to_name
from src.plotting import plot_similarity, get_similarities, main_correlation_to_shapley_baseline
from src.config_matplotlib import update_rcParams, clear_plot_figma

from src.experiments.attribution import load_attributions, root_attribution
from src.config.config import CalculateAttributions
from src.config.helpers import resolve_imagenet, load_all_configs, load_special_config
from src.explainers.helper_images import get_idx_filtered_n_superpixel


from src.interface import Attribution
from typing import List
from pathlib import Path


def main_attributions(cfg: CalculateAttributions):
    root = root_attribution(cfg, root=Path.cwd().parent)
    config_folders = load_all_configs(root=root)
    images, attributions = load_attributions(cfg, config_folders)
    visualize_attribution(images, attributions, title=f'{cfg.explainer.name}')


def main_segmentation(cfg: CalculateAttributions):
    """Visualize segmentation."""
    from src.experiments.resources import load_data
    from src.explainers.helper_images import load_segmentor
    from src.datasets.imagenet import preprocess_imagenet

    image_index = 5
    list_n_superpixels = [25]
    # image = images[image_index]
    for n_superpixel in list_n_superpixels[:]:
        print(n_superpixel)
        cfg.explainer.segmentation.n_superpixel = n_superpixel
        if cfg.explainer.segmentation.name == 'segment_anything' or True:
            idx_images = get_idx_filtered_n_superpixel(n_superpixel=n_superpixel, n_deviation=10)
            images = load_data(dataset=cfg.dataset, n_samples=10, idx_images=idx_images)
            images = [images[-3]]
        else:
            images = load_data(dataset=cfg.dataset, n_samples=3)
        for image in images:

            generate_patches_fn = load_segmentor(cfg.explainer.segmentation)
            segmentation = generate_patches_fn(image)
            image_np = preprocess_imagenet(image=image)

            plt.figure(f'{image.image_name} - n={np.max(segmentation)}', figsize=(5, 5))
            # plt.figure(f'{image.image_name}', figsize=(5, 5))
            # plt.figure()
            plt.imshow(image_np)
            plt.imshow(segmentation[0], cmap='prism', alpha=0.8)
            plt.axis(False)
            plt.tight_layout(pad=0.1)

# ======================================================================================================================
# =========================================      CONVERGENCE ANALYSIS        ===========================================
# ======================================================================================================================
def get_cfgs(cfg: CalculateAttributions, n_evals: List[int]) -> List[CalculateAttributions]:
    cfgs = []
    for n_eval in n_evals:
        cfg_tmp = deepcopy(cfg)
        cfg_tmp.explainer.n_eval = n_eval
        cfgs.append(cfg_tmp)
    return cfgs


def get_attributions(n_evals: List[int], cfg: CalculateAttributions) -> [List[Attribution], List[List[Attribution]]]:
    n_evals.sort()
    all_cfgs = get_cfgs(cfg=cfg, n_evals=n_evals)

    config_folders = load_all_configs(root=root_attribution(cfg))
    _, references = load_attributions(all_cfgs[-1], config_folders)

    list_attributions = [load_attributions(cfg_tmp, config_folders)[1] for cfg_tmp in all_cfgs[:-1]]
    return references, list_attributions


def main_convergence_analysis(cfg: CalculateAttributions):
    px = 2 / plt.rcParams['figure.dpi']  # pixel in inches
    if cfg.figma:
        figsize = (195 * px, 70 * px)
    else:
        figsize = (6, 3)
    # list_n_eval = [25, 100, 250, 750, 1000, 2500, 5000]
    # list_n_eval = [100, 250, 750, 2500, 5000]             # MNIST
    # list_n_eval = [10, 50, 100, 500, 750, 1000]               # imagenet

    cfg_shapley = deepcopy(cfg)
    cfg_shapley.explainer.cardinality_coalitions = None
    cfg_shapley.explainer.name = 'Shapley values'
    references_shapley, _ = get_attributions(n_evals=cfg.n_evals, cfg=cfg_shapley)

    # cardinalities = [-1, -5, -15, 25, 15, 5, 0]
    # cfg.cardinalities = [None]
    cfg_tmp = deepcopy(cfg)
    colors = plt.cm.viridis(np.linspace(0, 1, len(cfg.cardinalities)))

    fig = plt.figure('convergence', figsize=figsize)
    ax = fig.add_subplot()

    for i, s in enumerate(cfg.cardinalities + [None]):
        cfg_tmp.explainer.cardinality_coalitions = [s] if isinstance(s, int) else s

        references, list_attributions = get_attributions(n_evals=cfg.n_evals, cfg=cfg_tmp)
        numerical_fidelity, similarity, similarity_error = get_similarities(list_attributions, references,
                                                                            which='pearson')
        if s is None:
            plot_similarity(ax, numerical_fidelity, similarity, similarity_error,
                            label=map_attribution_to_name(list_attributions[0][0], depth=-1),
                            color='C2', marker='D')
        else:
            plot_similarity(ax, numerical_fidelity, similarity, similarity_error,
                            label=map_attribution_to_name(list_attributions[0][0], depth=-1),
                            color=colors[i])

    # plt.legend(ncol=int(len(cfg.cardinalities) / 3))
    if cfg.figma:
        clear_plot_figma()


def main_convergence_analysis_temp(cfg: CalculateAttributions):
    # list_n_eval = [25, 100, 250, 750, 1000, 2500, 5000]
    # list_n_eval = [10, 50, 100, 500, 750, 1000]               # imagenet

    # cfg_tmp = deepcopy(cfg)
    # for s in cardinalities:
    # cfg_tmp.explainer.cardinality_coalitions = [s] if isinstance(s, int) else s
    config_folders = load_all_configs(root=root_attribution(cfg))
    n_superpixel_list = [10, 25, 50, 75, 100]

    for n_superpixel in n_superpixel_list:
        cfg.explainer.segmentation.n_superpixel = n_superpixel
        all_cfgs_n_evals = get_cfgs(cfg=cfg, n_evals=cfg.n_evals)

        list_attributions = [load_attributions(cfg_tmp, config_folders)[1] for cfg_tmp in all_cfgs_n_evals]
        plot_similarity(list_attributions=list_attributions[:-1], references=list_attributions[-1],
                        label_depth=-1, which='pearson')

    plt.legend(n_superpixel_list, title='n_superpixel')


cs = ConfigStore.instance()
cs.store(name="config", node=CalculateAttributions)
OmegaConf.register_new_resolver('resolve_imagenet', resolve_imagenet)


@hydra.main(config_path='../src/conf', config_name='visualize_results', version_base=None)
def main(cfg: CalculateAttributions):
    print(OmegaConf.to_yaml(cfg))
    update_rcParams(fig_width_pt=234.88 * 0.85, half_size_image=True)

    if 'attributions' in cfg.plotting_routines:
        main_attributions(cfg)
    if 'segmentation' in cfg.plotting_routines:
        main_segmentation(cfg)
    if 'convergence_analysis' in cfg.plotting_routines:
        main_convergence_analysis(cfg)
        # main_convergence_analysis_temp(cfg)
    if 'correlation_to_shapley' in cfg.plotting_routines:
        px = 2 / plt.rcParams['figure.dpi']  # pixel in inches
        if cfg.figma:
            figsize = (195 * px, 70 * px)
        else:
            figsize = (6, 3)
        fig = plt.figure('cardinality_dependent_correlation_with_shapley' + f'_{cfg.figma}', figsize=figsize)
        ax = fig.add_subplot()
        for imputer_name in cfg.imputers:
            cfg.explainer.imputer = load_special_config(imputer_name, type='imputer')
            main_correlation_to_shapley_baseline(cfg, ax)

        if cfg.figma:
            clear_plot_figma()
        plt.legend()


if __name__ == '__main__':
    main()
