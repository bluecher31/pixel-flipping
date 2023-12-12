import numpy as np
import pickle
from skimage.segmentation import slic

from functools import partial
from pathlib import Path
from hydra.utils import get_original_cwd

from src.interface import Image
from typing import Union, Callable, List, Dict
from numpy.typing import NDArray



def _convert_image(image: Union[Image, NDArray]) -> NDArray:
    if isinstance(image, Image):
        image_np = image.image
    else:
        image_np = image
    assert image_np.ndim >= 2, 'Not implemented yet for 3D image.'
    return image_np


def generate_superpixel(image: Union[Image, NDArray], n_segments=100, compactness=20) -> NDArray:
    """Based on SLIC algorithm from skimage. Increase 'compactness' for more squared superpixels."""
    image_np = _convert_image(image=image)

    segmentation = slic(image_np, n_segments=n_segments, compactness=compactness, channel_axis=None)
    return segmentation


def create_dense_segmentation(sa_mask: List[Dict]) -> NDArray[int]:
    """Convert mask from SegmentAnything to a dens segmentation"""
    mask_dense = np.zeros((224, 224), dtype=np.int64)
    for i, m in enumerate(sa_mask):
        assert m['crop_box'] == [0, 0, 224, 224]
        mask_dense += m["segmentation"].astype(np.int64) * (i + 1)
    return mask_dense


def check_segmenation_anything(n_superpixel: int, n_deviation: int, segmentations: NDArray[int]) -> NDArray[bool]:
    n_superpixels = np.max(segmentations, axis=(-1, -2))
    n_min = n_superpixel - n_deviation
    n_max = n_superpixel + n_deviation
    return np.logical_and(n_superpixels > n_min, n_superpixels < n_max)


def get_idx_filtered_n_superpixel(n_superpixel: int, n_deviation: int) -> List[int]:
    """Get all idx which have n_superpixel in the right range. """

    path_to_files = Path(get_original_cwd()).parent / 'src' / 'segment_anything'
    with open(path_to_files / 'sam_imagenet_1000' / 'segmentation_masks_1000.npy', 'rb') as f:
        segmentations_1000 = np.load(f)

    mask_n = check_segmenation_anything(n_superpixel, n_deviation, segmentations_1000)
    idx_n = np.argsort(mask_n)[-np.sum(mask_n):]
    return sorted(list(idx_n))


def get_generate_segment_anything(cfg_segmentation) -> Callable:
    """Return pre-computed segmentations which are hand-picked from SegmentAnything. Only valid for 'Image' class."""
    path_to_files = Path(get_original_cwd()).parent / 'src' / 'segment_anything'
    assert path_to_files.exists()
    # with open(path_to_files / 'sam_masks_imagenet_selected', 'rb') as f:
    #     masks = pickle.load(f)

    with open(path_to_files / 'sam_imagenet_1000' / 'segmentation_masks_1000.npy', 'rb') as f:
        segmentations_1000 = np.load(f)

    # n_superpixels = np.max(segmentations_1000, axis=(1, 2))
    # with open(path_to_files / 'idx.npy', 'rb') as f:
    #     idx = np.load(f)

    with open(path_to_files / 'sam_imagenet_1000' / 'images_name_1000.npy', 'rb') as f:
        image_names = np.load(f)

    def generate_segment_anything(image: Union[Image, NDArray]) -> NDArray:
        assert isinstance(image, Image), 'Only valid for Image class, as we require access to image.image_name.'
        assert image.image_name in image_names, f'No segmentation exists for this image: {image.image_name}.'

        index_image = np.argmax(image_names == image.image_name)
        segmentation = segmentations_1000[index_image]
        mask_n = check_segmenation_anything(n_superpixel=cfg_segmentation.n_superpixel,
                                            n_deviation=cfg_segmentation.n_deviation,
                                            segmentations=segmentation)
        assert np.alltrue(mask_n), f'Something went wrong: {mask_n}'
        return np.stack([segmentation for _ in range(3)])  # shape: (3, *shape_image)
    # segmentation = create_dense_segmentation(sa_mask=masks[index_image])
    # list_segmentation = [create_dense_segmentation(mask) for mask in masks]
    # n_superpixels = [s.max() for s in list_segmentation]
    return generate_segment_anything


def patch_superpixel_vit(patch_size_input_space: int, image_shape=(224, 224)) -> NDArray:
    """Generates squared superpixels to match the internal patches of a vision transformer. """
    assert (image_shape[0] % patch_size_input_space == 0), f'{image_shape[0] % patch_size_input_space} != 0'
    assert (image_shape[1] % patch_size_input_space == 0), f'{image_shape[1] % patch_size_input_space} != 0'

    segmentation = np.zeros(image_shape, dtype=np.int64)

    Ly = image_shape[0] // patch_size_input_space
    Lx = image_shape[1] // patch_size_input_space

    for i in range(Lx * Ly):
        x = (i % Lx) * patch_size_input_space
        y = (i // Lx) * patch_size_input_space
        segmentation[y:y + patch_size_input_space, x:x + patch_size_input_space] = i
    return segmentation + 1


def harmonize_segmentations(seg_coarse: NDArray, seg_finegrained: NDArray) -> NDArray:
    finegrained_superpixels = list(np.unique(seg_finegrained))
    for i_coarse in np.unique(seg_coarse):
        mask_slic = seg_coarse == i_coarse
        for i_finegrained in finegrained_superpixels:
            mask_finegrained = seg_finegrained == i_finegrained
            n_pixels_finegrained = int(np.sum(mask_finegrained))
            mask_inside_slic = mask_finegrained[mask_slic]
            pixels_inside_slic_superpixels = np.sum(mask_inside_slic)
            if pixels_inside_slic_superpixels == n_pixels_finegrained:  # all pixels lie inside slic
                finegrained_superpixels.remove(i_finegrained)
            elif pixels_inside_slic_superpixels == 0:  # all pixels are outside
                pass
            elif n_pixels_finegrained > pixels_inside_slic_superpixels > n_pixels_finegrained / 2:
                raise NotImplementedError


def load_segmentor(cfg_segmentation) -> Callable[[NDArray], NDArray]:
    cfg = cfg_segmentation
    if cfg.name == 'slic':
        return partial(generate_superpixel, n_segments=cfg.n_superpixel, compactness=cfg.compactness_slic)
    elif cfg.name == 'patch_vit':
        assert cfg_segmentation.n_superpixel == cfg_segmentation.patch_size**2, \
            f'n_superpixels={cfg_segmentation.n_superpixel} is determined by internal ' \
            f'patch_size**2={cfg_segmentation.patch_size**2}'

        def generate_patches(image: Union[Image, NDArray]) -> NDArray:
            # image_np = _convert_image(image=image)
            patch_size_input_space = cfg.image_size / cfg.patch_size
            assert patch_size_input_space.is_integer(), 'patch_size and image_size do not match.'
            segmentation = patch_superpixel_vit(patch_size_input_space=int(patch_size_input_space),
                                                image_shape=(cfg.image_size, cfg.image_size))
            # shape: (image_shape), ndim = 2
            return np.stack([segmentation for _ in range(3)])       # shape: (3, *shape_image)
        return generate_patches
    elif cfg.name == 'patch_vit_slic':
        def generate_patches(image: Union[Image, NDArray]) -> NDArray:
            seg_slic_3 = generate_superpixel(image=image, n_segments=cfg.n_superpixel, compactness=cfg.compactness_slic)
            seg_slic = seg_slic_3[0]    # shape: image_size
            seg_vit = patch_superpixel_vit(patch_size_input_space=cfg.patch_size, image_shape=cfg.image_shape)
            # shape: (image_shape), ndim = 2

            segmentation = harmonize_segmentations(seg_coarse=seg_slic, seg_finegrained=seg_vit)

            return np.stack([segmentation for _ in range(3)])  # shape: (3, *shape_image)

        return generate_patches
    elif cfg.name == 'segment_anything':
        return get_generate_segment_anything(cfg_segmentation)
    else:
        raise ValueError(f'Segmentor not defined: {cfg.name}')
