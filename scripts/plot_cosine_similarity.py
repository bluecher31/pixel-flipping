import torch

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

from pathlib import Path
from src.config.helpers import load_all_configs, resolve_imagenet, load_special_config
from src.models.utils import ForwardHook

from sklearn.metrics.pairwise import cosine_similarity

from src.config.config import VisualizeOODScoresConfig
from src.experiments.resources import load_model, load_data
from src.experiments.imputations import load_imputations, root_imputations, extract_fixed_n_imputations, get_segmentation
from src.config_matplotlib import update_rcParams, line_with_shaded_errorband
from src.models.imagenet import downsample_to_patch_mask
from src.plotting import map_imputer_name

from src.interface import Image
from numpy.typing import NDArray
from typing import List, Dict
from functools import partial


def calculate_cosine_distance(f1, f2):
    # distance = np.sum(f1*f2, axis=axis) / np.linalg.norm(f1, axis=axis, keepdims=True) / np.linalg.norm(f2, axis=axis, keepdims=True)
    assert len(f1.shape) == 2 or len(f2.shape) == 2, 'Only 2-dimensional array valid. Similarity computed for axis=1.'
    assert f1.shape == f2.shape, 'Not identical shape.'
    cross_correlation = cosine_similarity(f1, f2)
    distance = np.diagonal(cross_correlation)
    return np.squeeze(distance)


def get_hidden_feature_activation_global(samples: List[NDArray], model, hook) -> NDArray:
    list_features = []
    for sample in samples:
        tensor = torch.tensor(sample, dtype=torch.float32)
        _ = model(tensor[None])
        features = hook.stored.clone()
        list_features.append(features.detach().numpy())
    return np.stack(list_features).squeeze()


def register_hook(model, name: str = '') -> ForwardHook:
    if 'ResNet50' in model.model_name or 'resnet50' in model.model_name:
        hook = ForwardHook(model.layer4)  # move layer selection to cfg
    elif 'Madry' in model.model_name:
        # register hook at last block of feature extractor
        if 'block' in name:
            index = int(name.strip('block_'))
            hook = ForwardHook(model.blocks[index])
        elif 'norm' == name:
            # or after layer norm
            # does not really matter for our purpose
            # (per sample spearman correlation between non-masked token before/after imputation)
            # but after has smaller values
            hook = ForwardHook(model.norm)

    else:
        raise RuntimeError(f'hook not registered for this model. ')
    return hook


def occluded_superpixels_into_tokens(segmentation: NDArray, image_size: int, patch_size: int) -> NDArray[bool]:
    """
    Converts a segmentation mask into the index list of tokens which corresponds to the internally removed tokens.
    True: used for prediction
    False: ignored
    """
    patch_mask = downsample_to_patch_mask(segmentation, image_size=image_size,
                                          patch_size=patch_size)
    return np.array(patch_mask.flatten(), dtype=bool)


def get_hidden_features_internal_imputer(cfg, model, hook: ForwardHook, images: List[Image],
                                         n_occluded_superpixels_list: List[int]) -> Dict[int, NDArray]:

    def get_internal_hidden_activations(image, segmentation) -> NDArray:
        model.occluded_model_fn(image.image, segmentation)
        features = hook.stored.clone()
        return np.squeeze(features.detach().numpy())

    all_features_occluded = {}
    for n_occluded_superpixels in n_occluded_superpixels_list:
        segmentations = [get_segmentation(cfg.segmentation, image, n_occluded_superpixels) for image in images]
        assert model.model_name == 'Madry_ViT', 'Only implemented for the madry ViT.'

        occluded_tokens = np.stack([occluded_superpixels_into_tokens(segmentation,
                                                                     image_size=cfg.dataset.image_size,
                                                                     patch_size=cfg.model.patch_size)
                                    for segmentation in segmentations])
        features_occluded = np.stack([get_internal_hidden_activations(image, segmentation)
                                      for image, segmentation in zip(images, segmentations)])
        # shape: (n_samples, n_tokes=197 - n_occluded_superpixels, n_embeddings=384), 197 = 14**2 + 1
        all_features_occluded[n_occluded_superpixels] = features_occluded
        all_features_occluded[f'{n_occluded_superpixels}_occluded_tokens'] = occluded_tokens

    return all_features_occluded


def get_feature_activations(model, hook, input_tensor: torch.tensor, mask_features: List[bool]) \
        -> [torch.tensor, torch.tensor]:
    patch_mask = np.array([True] + list(mask_features))
    output = torch.nn.functional.softmax(model(input_tensor, patch_mask=patch_mask))
    features = hook.stored.clone()
    return output, features


def calculate_features_dict(cfg) -> Dict:
    # prepare
    model = load_model(model_cfg=cfg.model, dataset_name=cfg.dataset.name)
    hook = register_hook(model, name=cfg.hook_name)
    get_hidden_feature_activation = partial(get_hidden_feature_activation_global, model=model, hook=hook)
    # store features for all imputers
    features_dict = {}

    for imputer_name in tqdm(cfg.imputer_names):
        if imputer_name == 'internal':
            n_possible_default = np.linspace(1, cfg.segmentation.n_superpixel, 10, dtype=int)
            images = load_data(dataset=cfg.dataset, n_samples=100, train=False, format_numpy=False)
            images = sorted(images, key=lambda image: image.image_name)
            features_dict['internal'] = get_hidden_features_internal_imputer(cfg, model, hook, images=images,
                                                                             n_occluded_superpixels_list=list(
                                                                                 n_possible_default))
        else:
            cfg.imputer = load_special_config(imputer_name, type='imputer')
            config_folders = load_all_configs(root=root_imputations(cfg, root=Path.cwd().parent / 'outputs'))
            images, list_imputations = load_imputations(cfg, config_folders)
            n_possible = [imps.n_occluded_superpixels for imps in list_imputations[0]]

            # loop over different occlusion fractions
            all_features_occluded = {}
            for n_occluded_superpixels in n_possible:
                imputations_fixed_n = extract_fixed_n_imputations(list_imputations,
                                                                  n_occluded_superpixels=n_occluded_superpixels)

                features_occluded = get_hidden_feature_activation(samples=[imp.imputation for imp in imputations_fixed_n])
                all_features_occluded[n_occluded_superpixels] = features_occluded
            features_dict[imputer_name] = all_features_occluded

    # store original (full) features
    features_original = get_hidden_feature_activation(samples=[img.image for img in images])
    features_dict['images'] = features_original
    return features_dict


def get_hidden_features_imputer(cfg) -> Dict:
    dir_feature_attr = Path().cwd().parent / 'outputs/imagenet'
    assert dir_feature_attr.exists()
    file_name = dir_feature_attr / f'{cfg.model.name}_hidden_features_{cfg.hook_name}.pickle'

    if cfg.compute_fresh is False:
        print('Load pre-computed hidden feature activations.')
        # pickle.load(open(file, 'rb'))
        features_dict = pickle.load(open(file_name, 'rb'))
        # with open(dir_feature_attr / f'hidden_features_dict.pickle', 'rb') as file:
        #     features_dict = pickle.load(file)
        return features_dict

    features_dict = calculate_features_dict(cfg)

    with open(file_name, 'wb') as file:
        pickle.dump(features_dict, file)
        print('Not saving results')

    return features_dict


def convert_features(f: NDArray, model_name: str) -> NDArray:
    if model_name == 'Madry_ViT':
        f_copy = f.copy()
        i_class_token = 0
        f_visual_tokens = np.delete(f_copy, 0, axis=i_class_token).reshape((14, 14, 384))
        f = np.transpose(f_visual_tokens, (2, 0, 1))
    else:
        # f1/f2.shape: (n_samples, n_features=2048, width=7, height=7) for ResNet50
        pass
    f_final = np.transpose(f.reshape(f.shape[0], -1), (1, 0))
    # shape: (n_tokens/independent_vectors, n_features)
    return f_final


def plot_correlation_imputers(cfg, features_dict, n_occluded_superpixels: int):
    n_imputers = len(cfg.imputer_names)
    correlation_matrix = np.zeros(shape=(n_imputers, n_imputers))

    for i1, imputer_name_1 in enumerate(cfg.imputer_names):
        for i2, imputer_name_2 in enumerate(cfg.imputer_names):
            features_imputer_1 = features_dict[imputer_name_1][n_occluded_superpixels]
            features_imputer_2 = features_dict[imputer_name_2][n_occluded_superpixels]
            if i1 == i2:
                features_imputer_2 = features_dict['images']

            # spatially resolved similarity per image
            cosine_distances = np.stack([calculate_cosine_distance(convert_features(f1, model_name=cfg.model.name),
                                                                   convert_features(f2, model_name=cfg.model.name),
                                                                   axis=0)      # corresponds to axis=1 for feature
                                         for f1, f2 in zip(features_imputer_1, features_imputer_2)])

            similarity_per_image = np.mean(cosine_distances, axis=(1, 2))

            correlation_matrix[i1, i2] = similarity_per_image.mean()
    df_correlation_matrix = pd.DataFrame(correlation_matrix, columns=cfg.imputer_names, index=cfg.imputer_names)

    scale = 2.

    plt.figure(f'cosine similarity - {cfg.model.name} - n={n_occluded_superpixels}', figsize=(scale * len(df_correlation_matrix.columns), scale + 0.34))
    plt.title(f'n_occluded_superpixels={n_occluded_superpixels}, {n_occluded_superpixels/cfg.segmentation.n_superpixel:.3f}%')
    sns.heatmap(df_correlation_matrix, annot=True, vmin=0, vmax=1,
                cmap=sns.color_palette("rocket_r", as_cmap=True), )
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout(pad=0.1)


def plot_similarity(df: pd.DataFrame, title: str, imputer_names: List[str], n_superpixel: int) -> plt.Axes:
    # plotting
    figsize = (4, 2)
    fig = plt.figure(title, figsize=figsize)
    ax = fig.add_subplot()
    for imputer_name in imputer_names:
        line_with_shaded_errorband(ax, x=df.index / n_superpixel, y=df[imputer_name], yerr=df[imputer_name + '_error'],
                                   label=map_imputer_name(imputer_name, 'name'), color=map_imputer_name(imputer_name, 'color'))
    plt.legend()
    plt.title(title)
    plt.xlabel('percentage occluded')
    plt.ylabel('cosine similarity')
    plt.tight_layout(pad=0.1)


def plot_occlusion_dependence(cfg, features_dict):
    # calculate cosine similarities
    imputer_names = [name for name in cfg.imputer_names if name != 'internal']
    init = False
    for imputer_name in imputer_names:
        if init is False:  # init df
            df = pd.DataFrame(index=features_dict[imputer_name].keys())
            df.index = df.index.set_names('n_superpixel')
            init = True

        mean = []
        error = []
        for n_occluded_superpixels in features_dict[imputer_name]:
            features_occluded_dict = features_dict[imputer_name]

            cosine_distances = np.stack([calculate_cosine_distance(convert_features(f1, model_name=cfg.model.name),
                                                                   convert_features(f2, model_name=cfg.model.name))      # corresponds to axis=1 for feature
                                         for f1, f2 in zip(features_dict['images'],     # loops over n_sample dimension
                                                           features_occluded_dict[n_occluded_superpixels])])

            similarity_per_image = np.mean(cosine_distances.reshape(cosine_distances.shape[0], -1), axis=(-1))
            mean.append(similarity_per_image.mean())
            error.append(similarity_per_image.std() / np.sqrt(len(similarity_per_image)))

        df = df.join(pd.DataFrame({imputer_name: mean, imputer_name + '_error': error},
                                  index=features_dict[imputer_name].keys()))

    plot_similarity(df=df, imputer_names=imputer_names, n_superpixel=cfg.segmentation.n_superpixel,
                    title=f'{cfg.model.name} - original features')


def filter_features(features: NDArray, class_token: bool,
                    mask_occluded_tokens: NDArray) -> NDArray:
    # the mask is always the same for all images since the segmentation mask is fixed.
    mask_token_global = np.concatenate((np.array([class_token]), mask_occluded_tokens[0]))
    assert len(mask_token_global) == features.shape[1], 'Number of tokens does not match. '
    assert mask_token_global.sum() > 0, 'Warning: removing all tokens!'

    features_remaining_tokens = np.compress(a=features, condition=mask_token_global, axis=1)
    return features_remaining_tokens


def get_features_remaining_tokens(features_dict: Dict[str, Dict], imputer_name: str, n: int,
                                  class_token: bool) -> NDArray:
    """Load features of remaining tokens in matching format for both internal as well as occlusion based features."""
    if imputer_name == 'internal':
        if n != 196:
            features_internal_imputer = features_dict['internal'][n][:, (1 - class_token):]
        else:
            assert class_token, 'Using the remaining class_token is necessary for this option. '
            features_internal_imputer = features_dict['internal'][n][:, None]

        return features_internal_imputer
    else:
        if imputer_name == 'images':
            features_raw = features_dict['images']
            # mask_tokens_images = np.ones_like(features_raw, dtype=bool)[:, 1:, 0]   # shape: (n_samples, n_tokens)
        else:
            features_raw = features_dict[imputer_name][n]
        mask_tokens_images = features_dict['internal'][f'{n}_occluded_tokens']
        features_remaining_tokens = filter_features(features_raw,
                                                    mask_occluded_tokens=mask_tokens_images,
                                                    class_token=class_token)
        return features_remaining_tokens


def plot_similarity_internal(cfg, features_dict: Dict[str, Dict[int, NDArray]]):
    imputer_names = [name for name in cfg.imputer_names if name != 'internal']
    init = False
    for imputer_name in imputer_names:
        if init is False:  # init df
            df = pd.DataFrame(index=features_dict[imputer_name].keys())
            df.index = df.index.set_names('n_superpixel')
            init = True

        mean = []
        error = []
        list_n = list(features_dict[imputer_name].keys())
        if cfg.segmentation.n_superpixel in list_n and cfg.class_token is False:
            list_n.remove(cfg.segmentation.n_superpixel)

        for n_occluded_superpixels in list_n:

            features_remaining_tokens = get_features_remaining_tokens(features_dict=features_dict,
                                                                      imputer_name=imputer_name,
                                                                      n=n_occluded_superpixels,
                                                                      class_token=cfg.class_token)

            features_internal_imputer = get_features_remaining_tokens(features_dict=features_dict,
                                                                      imputer_name='internal',
                                                                      n=n_occluded_superpixels,
                                                                      class_token=cfg.class_token)

            cosine_distances = np.stack([calculate_cosine_distance(f1, f2)  # corresponds to axis=1 for feature
                                         for f1, f2 in zip(features_internal_imputer,  # loops over n_sample dimension
                                                           features_remaining_tokens)])

            similarity_per_image = np.mean(cosine_distances.reshape(cosine_distances.shape[0], -1), axis=(-1))
            mean.append(similarity_per_image.mean())
            error.append(similarity_per_image.std() / np.sqrt(len(similarity_per_image)))

        df = df.join(pd.DataFrame({imputer_name: mean, imputer_name + '_error': error},
                                  index=list_n))

    plot_similarity(df=df, imputer_names=imputer_names, n_superpixel=cfg.segmentation.n_superpixel,
                    title=f'{cfg.model.name} - internal feature similarity')


def plot_similarity_to_remaining_tokens(cfg, features_dict: Dict[str, Dict[int, NDArray]]):
    imputer_names = [name for name in cfg.imputer_names]
    init = False

    get_features_remaining_tokens_local = partial(get_features_remaining_tokens,
                                                  features_dict=features_dict, class_token=cfg.class_token)

    for imputer_name in imputer_names:
        list_n = [n for n in features_dict[imputer_name].keys() if str(n).isdigit()]
        if cfg.segmentation.n_superpixel in list_n and cfg.class_token is False:
            list_n.remove(cfg.segmentation.n_superpixel)

        if init is False:  # init df
            df = pd.DataFrame(index=list_n)
            df.index = df.index.set_names('n_superpixel')
            init = True

        mean = []
        error = []
        for n_occluded_superpixels in list_n:
            features_remaining_tokens = get_features_remaining_tokens_local(imputer_name=imputer_name,
                                                                            n=n_occluded_superpixels)

            features_images = get_features_remaining_tokens_local(imputer_name='images', n=n_occluded_superpixels)

            cosine_distances = np.stack([calculate_cosine_distance(f1, f2)  # corresponds to axis=1 for feature
                                         for f1, f2 in zip(features_images,    # 197 # loops over n_sample dimension
                                                           features_remaining_tokens)])     # 196

            similarity_per_image = np.mean(cosine_distances.reshape(cosine_distances.shape[0], -1), axis=(-1))
            mean.append(similarity_per_image.mean())
            error.append(similarity_per_image.std() / np.sqrt(len(similarity_per_image)))

        df = df.join(pd.DataFrame({imputer_name: mean, imputer_name + '_error': error},
                                  index=list_n))

    plot_similarity(df=df, imputer_names=imputer_names, n_superpixel=cfg.segmentation.n_superpixel,
                    title=f'{cfg.model.name} - similarity remaining tokens')


cs = ConfigStore.instance()
cs.store(name="config", node=VisualizeOODScoresConfig)
OmegaConf.register_new_resolver('resolve_imagenet', resolve_imagenet)


@hydra.main(config_path='../src/conf', config_name='plot_cosine_similarity', version_base=None)
def main(cfg: VisualizeOODScoresConfig):
    update_rcParams(fig_width_pt=234.88 * 0.85, half_size_image=True)
    print(OmegaConf.to_yaml(cfg))
    features_dict = get_hidden_features_imputer(cfg)

    if 'feature_similarity_to_image' in cfg.plotting_routines:
        plot_occlusion_dependence(cfg, features_dict)

        # n_occluded_superpixels_visualize = [3, 6, 11, 17]
        # n_occluded_superpixels_visualize = [1, 22, 87, 131]
        # for n_occluded_superpixels in n_occluded_superpixels_visualize:
        #     plot_correlation_imputers(cfg, features_dict, n_occluded_superpixels)

    if 'similarity_to_internal' in cfg.plotting_routines:
        plot_similarity_internal(cfg, features_dict)

    if 'similarity_to_remaining_tokens' in cfg.plotting_routines:
        plot_similarity_to_remaining_tokens(cfg, features_dict)



if __name__ == '__main__':
    main()
