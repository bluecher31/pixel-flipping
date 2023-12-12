import copy

import pandas as pd
import torch

import hydra
from hydra.utils import get_original_cwd
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, DictConfig

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy, iqr
import pickle
from pathlib import Path
from src.config.helpers import compare_cfg, load_all_configs
from src.config.helpers import load_special_config
from executable_scripts.measure_ood_scores import root_ood

from src.interface import Measurement
from src.config.config import VisualizeOODScoresConfig
from src.datasets.imagenet import load_class_names
from src.config_matplotlib import update_rcParams, line_with_shaded_errorband
from src.plotting import map_imputer_name, calculate_auc
from numpy.typing import NDArray
from typing import List, Tuple, Dict


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def ood_score(model,tensor,probs=False):#the original Hendrycks OOD score uses probabilities
    #classic(Hendrycks et al) https://arxiv.org/abs/1610.02136
    #recent https://arxiv.org/abs/2110.06207
    out = model(tensor)
    if(probs):
        out = torch.nn.functional.softmax(out,dim=-1)
    return out.max(dim=-1)[0]


def class_entropy(model,tensor):
    # a la Madry https://arxiv.org/abs/2204.08945
    out = model(tensor)
    out = torch.nn.functional.softmax(out, dim=-1)
    return entropy(out.detach().numpy(), axis=-1)
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


def get_figsize(name: str) -> Tuple[float, float]:
    if name == 'paper_pdf':
        figsize = (3, 1.5)
    elif name == 'figma_one_third':
        figsize = (2., 1.5)
    elif name == 'figma_half_size':
        figsize = (2.5, 1.5)
    else:
        figsize = (6, 3)

    return figsize


def load_measurements(cfg: VisualizeOODScoresConfig, cardinality: int,
                      config_folders: List[Tuple[Path, DictConfig]]) -> List[List[Path]]:

    def compare_cfg_ood(cfg_base: DictConfig, cfg_test: DictConfig) -> bool:
        keys = ['dataset', 'segmentation', 'model', 'imputer']
        if compare_cfg(cfg_base, cfg_test, keys=keys, set_keys=['platform']) is False:
            return False
        return True

    folders = [path for (path, cfg_test) in config_folders if compare_cfg_ood(cfg, cfg_test)]
    # TODO: use select folder here
    #  but this breaks multiple occlusions for single sample (which needs access to different runs)
    folder_name = f'n_superpixel={cfg.segmentation.n_superpixel}, s={cardinality}'
    folders_specs = [folder / folder_name
                     for folder in folders]
    folders_specs = [folder for folder in folders_specs if folder.exists()]         # filter

    files_measurements = [[dir_image / 'occluded_prediction.pickle' for dir_image in folder.glob('*') if dir_image.is_dir()]
                          for folder in folders_specs]
    files_measurements_sorted = [sorted([file for file in files if file.exists()])
                                 for files in files_measurements]
    assert len(files_measurements_sorted) > 0, f'No files found for imputer: {cfg.imputer}.\n{folder_name}\n{cfg}'
    return files_measurements_sorted


def load_averaged_occluded_predictions(list_files: List[List[Path]], i_ref_files: int) -> [NDArray, int]:
    n_samples = len(list_files[i_ref_files])
    list_files_matching = [files for files in list_files if len(files) == n_samples]
    # assert len(list_files_matching) > 1, 'Nothing to average over. '
    all_predictions = []
    for files in list_files_matching:
        occluded_predictions = np.stack([pickle.load(open(file, 'rb')) for file in files])
        all_predictions.append(occluded_predictions)
    all_predictions_np = np.stack(all_predictions)
    return np.mean(all_predictions_np, axis=0), len(list_files_matching)


def load_different_samples(cfg: VisualizeOODScoresConfig, cardinality: int) \
        -> [NDArray, NDArray]:
    root = root_ood(dataset_name=cfg.dataset.name, model_name=cfg.model.name, imputer_name=cfg.imputer.name)
    config_folders = load_all_configs(root=root)
    list_files = load_measurements(cfg=cfg, cardinality=cardinality, config_folders=config_folders)

    index_files = -1        # corresponds to newest since list_files is sorted
    if cfg.average_over_imputations is False:
        occluded_predictions = np.stack([pickle.load(open(file, 'rb')) for file in list_files[index_files]])
    else:
        occluded_predictions, n_samples = load_averaged_occluded_predictions(list_files, -1)
        if cardinality == 1:
            print(f'For {cfg.imputer.name} averaged over {n_samples} samples.')

    if (list_files[index_files][0].parent / 'actual_fraction_occluded.pickle').exists():
        actual_fraction_occluded = np.stack([pickle.load(open(file.parent / 'actual_fraction_occluded.pickle', 'rb'))
                                             for file in list_files[index_files]])
    else:
        # if cardinality == 1:
        #     print('WARNING: calculate fraction occluded based on the number of occluded superpixels. ')
        actual_fraction_occluded = np.repeat(cardinality / cfg.segmentation.n_superpixel, len(occluded_predictions))

    assert len(occluded_predictions) == cfg.n_samples, \
        f'Loaded number of samples={len(occluded_predictions)} does not match requested number={cfg.n_samples}. \n' \
        f'{cfg}'
    return np.squeeze(occluded_predictions), actual_fraction_occluded


def load_multiple_occlusions_for_unique_sample(cfg: VisualizeOODScoresConfig, image_name: str, cardinality: int,
                                               imputer_name: str) -> NDArray:
    root = root_ood(dataset_name=cfg.dataset.name, model_name=cfg.model.name, imputer_name=cfg.imputer.name)
    config_folders = load_all_configs(root=root)
    list_files = load_measurements(cfg=cfg, cardinality=cardinality, config_folders=config_folders)
    files_image = [file for files in list_files for file in files
                   if file.parent.name == image_name]
    assert len(files_image) > 0, f'No occluded_predictions found for image: {image_name}'
    occluded_predictions = np.stack([pickle.load(open(file, 'rb')) for file in files_image])
    return occluded_predictions


def cosine_similarity_array(arr1: NDArray, arr2: NDArray) -> NDArray:
    """Calculate cosine similarity between two arrays (heatmaps)."""
    a = arr1.reshape(arr1.shape[0], -1)
    b = arr2.reshape(arr2.shape[0], -1)
    a_times_b = a * b
    cos_sim = a_times_b.sum(axis=-1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1)
    return cos_sim


def average_measurements(values: NDArray) -> Measurement:
    mean = np.mean(values)
    error = np.std(values) / np.sqrt(len(values))
    measurement = Measurement(values=mean, error=error)
    return measurement


def calculate_ood_score(occluded_predictions: NDArray, which: str, **kwargs) -> Measurement:
    if which == 'ood':
        scores = occluded_predictions.max(-1)
    elif which == 'entropy':
        scores = entropy(occluded_predictions, axis=-1)
    elif which == 'label':
        labels = kwargs['labels']
        assert len(labels) == len(occluded_predictions), f'Something went wrong: {occluded_predictions.shape}'
        scores = np.stack([prediction[label] for prediction, label in zip(occluded_predictions, labels)])
    elif which == 'correlation_baseline':
        baseline_occluded_predictions = kwargs['baseline_occluded_predictions']     # shape: (n_samples, n_classes)
        scores = cosine_similarity_array(arr1=occluded_predictions, arr2=baseline_occluded_predictions)
        # scores = np.abs(baseline_occluded_predictions - occluded_predictions).mean(-1)      # shape: (n_samples,)
    else:
        raise NotImplementedError(f'Not yet available: {which}')

    ood_score = average_measurements(values=scores)
    return ood_score


def plot_occluded_prediction(cfg, ax, label: str, **kwargs_plot) -> Dict:
    """Calculates and returns the AUC."""
    cardinalities = list(np.linspace(1, cfg.segmentation.n_superpixel, 10, dtype=int))
    ood_scores = []
    x_values = []
    kwargs = {}
    for cardinality in sorted(cardinalities):
        occluded_predictions, actual_fraction_occluded = load_different_samples(cfg=cfg, cardinality=cardinality)
        if cardinality == min(cardinalities) and cfg.which_ood_measure == 'label':
            kwargs['labels'] = occluded_predictions.argmax(-1)
        elif cfg.which_ood_measure == 'correlation_baseline':
            # if cardinality == min(cardinalities)
            cfg_tmp = copy.deepcopy(cfg)
            cfg_tmp.imputer = load_special_config('internal', type='imputer')
            baseline_occluded_predictions, _ = load_different_samples(cfg=cfg_tmp, cardinality=cardinality)
            kwargs['baseline_occluded_predictions'] = baseline_occluded_predictions

        ood_score = calculate_ood_score(occluded_predictions, which=cfg.which_ood_measure, **kwargs)
        ood_scores.append(ood_score)
        x_values.append(average_measurements(actual_fraction_occluded))
    y = np.stack([m.values for m in ood_scores])
    yerr = np.stack([m.error for m in ood_scores])
    x = np.stack([m.values for m in x_values])
    # plt.errorbar(x=cardinalities, y=y, yerr=yerr, label=imputer_name)
    line_with_shaded_errorband(ax=ax, x=x,
                               y=y, yerr=yerr, label=label, **kwargs_plot)

    auc = calculate_auc(x, y, xmin=min(x), xmax=max(x))

    ax.set_xlabel(r'Occlusion fraction')
    ax.legend()

    if cfg.plt_style == 'paper_pdf':
        ax.set_ylabel(f'R-OMS')
        ax.get_legend().remove()
    else:
        ax.set_ylabel(f'Score - {cfg.which_ood_measure}')
    plt.tight_layout(pad=0.1)
    slic_param = cfg.segmentation.compactness_slic if cfg.segmentation.name == 'slic' else None
    dict_auc = {'auc': auc, 'imputer': cfg.imputer.name,
                'n_superpixel': cfg.segmentation.n_superpixel, 'segmentor': cfg.segmentation.name,
                'model': cfg.model.name, 'compactness_slic': slic_param}
    return dict_auc


def plot_ood(cfg: VisualizeOODScoresConfig):

    figsize = get_figsize(name=cfg.plt_style)
    fig = plt.figure(f'{cfg.which_ood_measure} - {cfg.dataset.name} - {cfg.segmentation.n_superpixel} - {cfg.model.name}',
                     figsize=figsize)
    ax = fig.subplots()



    for imputer_name in cfg.imputer_names:
        cfg.imputer = load_special_config(imputer_name, type='imputer')
        plot_occluded_prediction(cfg, ax, label=map_imputer_name(imputer_name, what='name'),
                                 color=map_imputer_name(imputer_name, what='color'))

    if cfg.plt_style not in ['paper_pdf', 'figma_half_size']:
        plt.title(f'{cfg.model.name} - $n_{{superpixel}} = {cfg.segmentation.n_superpixel}$')


def plot_ood_superpixels(cfg: VisualizeOODScoresConfig):
    fig = plt.figure(f'imputer={cfg.imputer_name}_{cfg.which_ood_measure} - {cfg.dataset.name} - {cfg.model.name}', figsize=(5, 2.8))
    ax = fig.subplots()
    cfg.imputer = load_special_config(cfg.imputer_name, type='imputer')

    colors = plt.cm.viridis(np.linspace(0, 1, len(cfg.n_superpixels)))
    for i, n_superpixel in enumerate(sorted(cfg.n_superpixels)):
        cfg.segmentation.n_superpixel = n_superpixel
        plot_occluded_prediction(cfg, ax, label=n_superpixel, c=colors[i])

    plt.title(cfg.imputer.name)
    plt.legend(title='$n_{superpixel}$')


def plot_ood_superpixels_shape(cfg: VisualizeOODScoresConfig, n_superpixel: int = 75) -> pd.DataFrame:
    df_shape = pd.DataFrame()
    fig = plt.figure(f'superpixel_shape-imputer={cfg.imputer_name}_{cfg.which_ood_measure} - {cfg.dataset.name} - {cfg.model.name}', figsize=(5, 2.8))
    cfg.imputer = load_special_config(cfg.imputer_name, type='imputer')
    ax = fig.subplots()

    colors = plt.cm.viridis(np.linspace(0, 1, 3))
    compactness = [0.1, 1., 10.]
    cfg.segmentation = load_special_config('slic', type='segmentation')
    cfg.segmentation.n_superpixel = n_superpixel
    for i, c_slic in enumerate(compactness):
        cfg.segmentation.compactness_slic = c_slic
        dict_auc = plot_occluded_prediction(cfg, ax, label=f'slic={c_slic}', c=colors[i])
        df_new = pd.DataFrame(dict_auc, index=[0])
        df_shape = pd.concat([df_shape, df_new], ignore_index=True).sort_values(by='auc')

    cfg.segmentation = load_special_config('segment_anything', type='segmentation')
    cfg.segmentation.n_superpixel = n_superpixel
    dict_auc = plot_occluded_prediction(cfg, ax, label='SegmentAnything')
    df_new = pd.DataFrame(dict_auc, index=[0])
    df_shape = pd.concat([df_shape, df_new], ignore_index=True).sort_values(by='auc')

    plt.title(cfg.imputer.name)
    plt.legend(title='$n_{superpixel}$'+f'={n_superpixel}')
    return df_shape


def plot_ood_diffusion(cfg: VisualizeOODScoresConfig, which: str):
    cardinalities = list(np.linspace(1, cfg.segmentation.n_superpixel, 10, dtype=int))
    fig = plt.figure(f'{which} - diffusion - {cfg.dataset.name} - {cfg.segmentation.n_superpixel} - {cfg.model.name}', figsize=(5, 2.3))
    ax = fig.subplots()
    imputer_name = 'diffusion'
    n_resampling_list = [3, 5, 7, 8]
    for n_resampling in n_resampling_list:
        path_to_imputer_config = Path(get_original_cwd()).parent / f'src/conf/imputer/{imputer_name}.yaml'
        imputer_conf = OmegaConf.load(path_to_imputer_config)
        cfg.imputer = imputer_conf
        cfg.imputer.n_resampling = n_resampling

        ood_scores = []
        for cardinality in cardinalities:
            occluded_predictions = load_different_samples(cfg=cfg, cardinality=cardinality)
            ood_score = calculate_ood_score(occluded_predictions, which=which)
            ood_scores.append(ood_score)
        y = np.stack([m.values for m in ood_scores])
        yerr = np.stack([m.error for m in ood_scores])
        # plt.errorbar(x=cardinalities, y=y, yerr=yerr, label=imputer_name)
        line_with_shaded_errorband(ax=ax, x=np.stack(cardinalities), y=y, yerr=yerr, label=str(n_resampling))
    plt.xlabel(r'Cardinality $s$')
    plt.ylabel(f'Score - {which}')
    plt.legend(title='$n_{resampling}$')
    plt.tight_layout(pad=0.1)


def visualize_ood_bias(cfg: VisualizeOODScoresConfig, imputer_name: str, cardinalities: List[int], image_name: str):
    class_names = load_class_names()
    # fix a single image
    # i_image = 0
    fig = plt.figure(f'{image_name} imputer={imputer_name} n_superpixel={cfg.segmentation.n_superpixel}')
    ax = fig.subplots()
    ax.set_title(f'imputer: {imputer_name}, $n_{{superpixel}}={cfg.segmentation.n_superpixel}$')
    # FUTURE: loop cardinalities
    # load multiple occluded predictions for a fixed cardinality
    for cardinality in cardinalities:
        occluded_predictions = load_multiple_occlusions_for_unique_sample(
            cfg=cfg, cardinality=cardinality, imputer_name=imputer_name, image_name='ILSVRC2012_val_00027830')

        # calculate average prediction for occluded images
        average_prediction = occluded_predictions.mean(axis=0)
        # visualize leading classes
        top_k_classes = average_prediction.argsort()[-10:][::-1]

        for i, [prediction, class_number] in enumerate(zip(average_prediction[top_k_classes], top_k_classes)):
            if i < 3:
                marker = '$' + str(class_names[class_number]) + '$'
                s = 1000
            else:
                marker = '$' + str(class_number) + '$'
                s = 250
            ax.scatter(cardinality, prediction, marker=marker, s=s)

    ax.set_xlabel('Cardinality $s$')
    ax.set_ylabel('average prediction')
    plt.tight_layout(pad=0.1)


def main_visualize_ood_bias(cfg: VisualizeOODScoresConfig, image_name='ILSVRC2012_val_00027830'):
    cardinalities = np.linspace(1, cfg.segmentation.n_superpixel, 10, dtype=int)
    for imputer_name in cfg.imputer_names:
        visualize_ood_bias(cfg=cfg, imputer_name=imputer_name, cardinalities=list(cardinalities), image_name=image_name)


def load_ood_auc_results(cfg) -> pd.DataFrame:
    file_path = Path().cwd().parent / 'imagenet' / f'ood_{cfg.model.name}.pickle'
    assert file_path.parent.exists()
    cfg.filter_images = False
    if cfg.compute_df_fresh is False:
        print('Load pre-computed local OMS.')
        df = pickle.load(open(file_path, 'rb'))
        return df

    models = ['timm', 'madry_vit']
    imputers = ['cv2', 'trainset', 'constant_value', 'color_histogram']
    n_superpixels = [25, '75_square', '75_segment', 500]

    df_timm_vit = pd.DataFrame()
    for model in models:
        cfg.model = load_special_config(name=model, type='model')
        cfg.imputer = load_special_config('diffusion', 'imputer')
        cfg.segmentation = load_special_config('slic', 'segmentation')
        cfg.segmentation.n_superpixel = 200

        fig, ax = plt.subplots(1, 1)
        dict_auc = plot_occluded_prediction(cfg, ax, label='')
        plt.close(fig)

        df_new = pd.DataFrame(dict_auc, index=[0])
        df_timm_vit = pd.concat([df_timm_vit, df_new], ignore_index=True).sort_values(by='auc')

        for imputer in imputers:
            cfg.imputer = load_special_config(imputer, 'imputer')
            for n_superpixel in n_superpixels:
                if n_superpixel == '75_square':
                    cfg.segmentation = load_special_config('slic_squares', 'segmentation')
                    n_superpixel = 75
                elif n_superpixel == '75_segment':
                    cfg.segmentation = load_special_config('segment_anything', 'segmentation')
                    n_superpixel = 75
                else:
                    cfg.segmentation = load_special_config('slic', 'segmentation')
                cfg.segmentation.n_superpixel = n_superpixel
                fig, ax = plt.subplots(1, 1)
                dict_auc = plot_occluded_prediction(cfg, ax, label='')
                plt.close(fig)

                df_new = pd.DataFrame(dict_auc, index=[0])
                df_timm_vit = pd.concat([df_timm_vit, df_new], ignore_index=True).sort_values(by='auc')

    cfg.model = load_special_config('resnet50', 'model')

    df_shape = pd.DataFrame()
    for imputer_name in cfg.imputer_names:
        cfg.imputer_name = imputer_name
        df_new = plot_ood_superpixels_shape(cfg=cfg)
        df_shape = pd.concat([df_shape, df_new], ignore_index=True).sort_values(by='auc')
        plt.close('all')

    n_superpixels = [10, 25, 50, 100, 200, 500]
    cfg.segmentation = load_special_config(name='slic', type='segmentation')
    df_auc = pd.DataFrame()
    for n_superpixel in n_superpixels:
        cfg.segmentation.n_superpixel = n_superpixel
        for imputer_name in cfg.imputer_names:
            cfg.imputer = load_special_config(imputer_name, type='imputer')
            fig, ax = plt.subplots(1, 1)
            dict_auc = plot_occluded_prediction(cfg, ax, label=imputer_name)
            plt.close(fig)
            df_new = pd.DataFrame(dict_auc, index=[0])
            df_auc = pd.concat([df_auc, df_new], ignore_index=True).sort_values(by='auc')

    df_final = pd.concat([df_auc, df_shape, df_timm_vit], ignore_index=True).sort_values(by='auc')
    with open(file_path, 'wb') as file:
        print(f'Store ood at {file_path}.')
        pickle.dump(df_final, file)
    return df_final


def get_marker_color(param_slic) -> [str, str]:
    if param_slic == 0.1:           # very flexible
        return '*', '#9864ae'
    elif param_slic == 1.:
        return 'o', '#4c72b0'       # matches C0 in my standard cycle
    elif param_slic == 10.:         # squares
        return 's', '#414756'
    elif param_slic is None or np.isnan(param_slic):
        return 'X', 'C2'


def set_yaxis_auc(ax):
    yticks = [0.1, 0.3, 0.5]
    ax.set_yticks(yticks)
    ax.set_ylim(0.05, 0.58)
    # ax.set_ylabel('AUC[R-OMS]')


def analyze_varaiance_experiment(df_results: pd.DataFrame):
    list_max_diff = []
    for x in df_results:
        max_diff = df_results[x].max() - df_results[x].min()
        list_max_diff.append(max_diff)
    mean = np.mean(list_max_diff)
    max = np.max(list_max_diff)
    var_iqr = iqr(df_results, axis=0)
    dict_analysis = {'mean': mean, 'max': max}
    print(df_results.index.name)
    # print(dict_analysis)
    print(f'max_diff analysis: mean = {mean:.3f}    -    max = {max:.3f}      -      iqr = {iqr(list_max_diff):.3f}')
    print(f'iqr: mean = {var_iqr.mean():.3f}    -    max = {var_iqr.max():.3f}      -      iqr = {iqr(var_iqr):.3f}\n')


def get_auc_superpixel_n(df: pd.DataFrame, imputers: List[str], n_superpixels: List[int]) -> pd.DataFrame:
    dict_results = {}
    for i_imputer, imputer in enumerate(imputers):
        df_imputer = df[df['imputer'] == imputer]
        list_auc = []
        for i_superpixel, n_superpixel in enumerate(n_superpixels):
            df_point = df_imputer[df_imputer['n_superpixel'] == n_superpixel]
            assert len(df_point) == 1, 'More than one auc measurement.'
            list_auc.append(float(df_point['auc']))
        dict_results[imputer] = list_auc
    df_result = pd.DataFrame(dict_results, index=n_superpixels)
    df_result.index.name = 'n_superpixel'
    return df_result.sort_index()


def scatter_plot_auc_n(cfg, df: pd.DataFrame):
    imputers = np.unique(df['imputer'])
    imputers = np.array(sorted(imputers))[[1, 2, 0, 3, 4]]
    n_superpixels = sorted(np.unique(df['n_superpixel']))
    n_superpixels.remove(75)
    df_model = df[df['model'] == 'ResNet50']
    df_results = get_auc_superpixel_n(df=df_model, imputers=imputers, n_superpixels=n_superpixels)
    analyze_varaiance_experiment(df_results)

    figsize = get_figsize(name=cfg.plt_style)
    fig, ax = plt.subplots(1, 1, figsize=figsize, num='ood_scores_superpixels_n')
    colors = plt.cm.viridis(np.linspace(0, 1, len(df_results)))
    for i_imputer, imputer in enumerate(df_results):
        df_imputer = df_results[imputer]
        for i_superpixel, auc in enumerate(df_imputer):
            ax.scatter(i_imputer, auc, color=colors[i_superpixel], s=35)
    set_yaxis_auc(ax)
    ax.set_xticks(np.arange(len(imputers)), [map_imputer_name(imputer, 'name') for imputer in imputers], rotation=45)
    plt.tight_layout(pad=0.1)


def filter_shape(df: pd.DataFrame, compactness_slic: [float, type(None)]) -> pd.DataFrame:
    if compactness_slic is not None:
        df_exp = df[df['compactness_slic'] == compactness_slic]
    else:
        df_exp = df[df['segmentor'] == 'segment_anything']
    return df_exp


def get_auc_superpixel_shape(df: pd.DataFrame, imputers: List[str], list_compactness_slic: List[float]) -> pd.DataFrame:
    df_shape = df[df['n_superpixel'] == 75]
    dict_results = {}
    for i_imputer, imputer in enumerate(imputers):
        # if imputer == 'diffusion':
        #     continue
        df_imputer = df_shape[df_shape['imputer'] == imputer]
        list_auc = []
        for i_shape, param in enumerate(list_compactness_slic):
            df_point = filter_shape(df_imputer, compactness_slic=param)
            assert len(df_point) == 1, 'More than one auc measurement.'
            list_auc.append(float(df_point['auc']))
        dict_results[imputer] = list_auc

    df_result = pd.DataFrame(dict_results, index=list_compactness_slic)
    df_result.index.name = 'segmentor'
    return df_result.sort_index()


def scatter_plot_auc_superpixel_shape(cfg, df: pd.DataFrame):
    imputers = np.unique(df['imputer'])
    imputers = np.array(sorted(imputers))[[1, 2, 0, 3, 4]]
    n_superpixels = sorted(np.unique(df['n_superpixel']))
    n_superpixels.remove(75)
    df_model = df[df['model'] == 'ResNet50']
    slic_param = [10., 1., 0.1, None]
    df_results = get_auc_superpixel_shape(df=df_model, imputers=imputers, list_compactness_slic=slic_param)
    analyze_varaiance_experiment(df_results)

    figsize = get_figsize(name=cfg.plt_style)

    fig, ax = plt.subplots(1, 1, figsize=figsize, num='ood_scores_superpixels_shape')
    for i_imputer, imputer in enumerate(df_results):
        df_imputer = df_results[imputer].iloc[[2, 1, 0, 3]]

        for i_shape, auc in enumerate(df_imputer):
            marker, color = get_marker_color(df_imputer.index[i_shape])
            # offset = 0.035 * i_shape - 2 * 0.035
            offset = 0
            ax.scatter(i_imputer + offset, auc, s=25, marker=marker, c=color)
        set_yaxis_auc(ax)
        ax.set_xticks(np.arange(len(imputers)), [map_imputer_name(imputer, 'name') for imputer in imputers],
                      rotation=45)
        plt.tight_layout(pad=0.1)


def get_experiment_imputers(name: [str, int], df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(name, int):           # filter for number of standard slic superpixels
        df_exp = df[df['n_superpixel'] == name]
    elif name == 'square':
        df_exp = filter_shape(df, compactness_slic=10)
    elif name == 'normal':
        df_tmp = filter_shape(df, compactness_slic=1)
        df_exp = df_tmp[df_tmp['n_superpixel'] == 75]
    elif name == 'segment_anything':
        df_exp = filter_shape(df, compactness_slic=None)
    else:
        raise ValueError(f'This experiment is not defined: name = {name}')
    return df_exp


def get_auc_imputers(df: pd.DataFrame, x_experiment: List, y_imputers: List[str]) -> pd.DataFrame:
    dict_results = {}
    for i_name, name in enumerate(x_experiment):
        df_exp = get_experiment_imputers(name, df)
        list_auc = []
        for i_imputer, imputer in enumerate(y_imputers):
            df_point = df_exp[df_exp['imputer'] == imputer]
            if len(df_point) == 0:
                list_auc.append(np.nan)
            else:
                list_auc.append(float(df_point['auc']))
        dict_results[name] = list_auc
    df_result = pd.DataFrame(dict_results, index=df_exp['imputer'])
    df_result.index.name = 'imputers'
    return df_result.sort_index()


def map_name_imputer_exp(name: str) -> str:
    if isinstance(name, int):           # filter for number of standard slic superpixels
        return f'n = {name}'
    elif name == 'square':
        return 'Squares'
    elif name == 'normal':
        return 'Normal'
    elif name == 'segment_anything':
        return 'Segment\nAnything'
    else:
        raise ValueError(f'This experiment is not defined: name = {name}')


def scatter_plot_imputer(cfg, df: pd.DataFrame):
    imputers = np.unique(df['imputer'])
    imputers = np.array(sorted(imputers))[[1, 2, 0, 3, 4]]
    n_superpixels = sorted(np.unique(df['n_superpixel']))
    n_superpixels.remove(75)
    df_model = df[df['model'] == 'ResNet50']

    # select experiments
    names = [10, 'square', 'normal', 'segment_anything', 500]
    df_results = get_auc_imputers(df=df_model, x_experiment=names, y_imputers=imputers)
    analyze_varaiance_experiment(df_results)

    figsize = get_figsize(name=cfg.plt_style)
    fig, ax = plt.subplots(1, 1, figsize=figsize, num='ood_scores_imputers')
    for i_name, name in enumerate(df_results):
        df_exp = df_results[name]
        for i_imputer, auc in enumerate(df_exp):
            ax.scatter(i_name, auc, color=map_imputer_name(df_exp.index[i_imputer], what='color'), s=35)

    set_yaxis_auc(ax)
    ax.set_xticks(np.arange(len(names)), [map_name_imputer_exp(name) for name in names], rotation=45)
    plt.tight_layout(pad=0.1)


def get_experiment_model(name: str, df: pd.DataFrame) -> pd.DataFrame:
    n_superpixel, description = name.split('_')
    assert n_superpixel.isdigit()
    df_n = df[df['n_superpixel'] == int(n_superpixel)]

    if description in ['TrainSet', 'ConstantValueImputer', 'diffusion', 'ColorHistogram', 'cv2']:        # normal slic
        df_tmp = df_n[df_n['compactness_slic'] == 1.]
        df_exp = df_tmp[df_tmp['imputer'] == description]
    elif description == 'squares':
        df_tmp = df_n[df_n['compactness_slic'] == 10.]
        df_exp = df_tmp[df_tmp['imputer'] == 'cv2']
    elif description == 'segmentanything':
        df_tmp = df_n[df_n['segmentor'] == 'segment_anything']
        df_exp = df_tmp[df_tmp['imputer'] == 'cv2']
    else:
        raise ValueError(f'This experiment is not defined: name = {name}')
    assert len(df_exp) == 3, f'Expected three different models but only found {len(df_exp)}: {list(df_exp["model"])}\n{description}'
    return df_exp


def get_auc_models(df: pd.DataFrame, x_experiment: List, y_models: List[str]) -> pd.DataFrame:
    dict_results = {}
    for i_name, name in enumerate(x_experiment):
        df_exp = get_experiment_model(name, df)
        list_auc = []
        for i_model, model in enumerate(y_models):
            df_point = df_exp[df_exp['model'] == model]
            if len(df_point) == 0:
                list_auc.append(np.nan)
            else:
                list_auc.append(float(df_point['auc']))
        dict_results[name] = list_auc
    df_result = pd.DataFrame(dict_results, index=y_models)
    df_result.index.name = 'models'
    return df_result.sort_index()


def scatter_plot_models(cfg, df: pd.DataFrame):
    models = np.unique(df['model'])
    models = np.array(sorted(models))[[1, 2, 0]]

    # select experiments
    names = ['25_TrainSet', '75_squares', '75_segmentanything', '500_ConstantValueImputer', '200_diffusion']        # , '200_diffusion'
    df_results = get_auc_models(df=df, x_experiment=names, y_models=models)
    analyze_varaiance_experiment(df_results)
    figsize = get_figsize(name='large')
    fig, ax = plt.subplots(1, 1, figsize=figsize, num='ood_scores_models')
    for i_name, name in enumerate(df_results):
        df_exp = df_results[name]
        for i_model, model in enumerate(models):
            auc = df_exp[model]
            if i_name == 0:
                ax.scatter(i_name, auc, color=f'C{i_model}', s=35, label=model)
            else:
                ax.scatter(i_name, auc, color=f'C{i_model}', s=35)

    set_yaxis_auc(ax)
    ax.set_xticks(np.arange(len(names)), names, rotation=45)
    plt.legend()
    plt.tight_layout(pad=0.1)


cs = ConfigStore.instance()
cs.store(name="config", node=VisualizeOODScoresConfig)


@hydra.main(config_path='../src/conf', config_name='plot_ood', version_base=None)
def main(cfg: VisualizeOODScoresConfig):
    update_rcParams(fig_width_pt=234.88 * 0.85, half_size_image=True)
    print(OmegaConf.to_yaml(cfg))

    if 'ood' in cfg.plotting_routines:
        plot_ood(cfg=cfg)
    if 'ood_superpixels' in cfg.plotting_routines:
        for n_superpixel in cfg.n_superpixels:
            cfg.segmentation.n_superpixel = n_superpixel
            plot_ood(cfg=cfg)
    if 'ood_imputers' in cfg.plotting_routines:
        for imputer_name in cfg.imputer_names:
            cfg.imputer_name = imputer_name
            plot_ood_superpixels(cfg=cfg)
    if 'ood_shape_segmentanything' in cfg.plotting_routines:
        for imputer_name in cfg.imputer_names:
            cfg.imputer_name = imputer_name
            plot_ood_superpixels_shape(cfg=cfg)
    if 'ood_diffusion' in cfg.plotting_routines:
        plot_ood_diffusion(cfg=cfg, which='ood')

    if 'visualize_ood_bias' in cfg.plotting_routines:
        main_visualize_ood_bias(cfg)

    if 'plot_df' in cfg.plotting_routines:
        df = load_ood_auc_results(cfg)
        scatter_plot_models(cfg, df)
        scatter_plot_auc_n(cfg, df)
        scatter_plot_auc_superpixel_shape(cfg, df)
        scatter_plot_imputer(cfg, df)


if __name__ == '__main__':
    main()
