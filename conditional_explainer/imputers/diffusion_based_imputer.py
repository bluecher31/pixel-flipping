import torch
import numpy as np

import cv2

from src.diffusion_imputer.guided_diffusion.unet import UNetModel

from pathlib import Path

from conditional_explainer.imputers.abstract_imputer import Imputer

from src.diffusion_imputer import conf_mgt
from src.diffusion_imputer.utils import yamlread
from src.diffusion_imputer.guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    #classifier_defaults,
    create_model_and_diffusion,
    #create_classifier,
    create_cond_fn,
    select_args,
) 
from src.datasets.imagenet import(
    UnNormalize,
    imagenet_norm
)
from torchvision.transforms import Normalize

from typing import Tuple, Any, Callable
from numpy.typing import NDArray


def make_center_mask(shape_mask: Tuple, startx: int, starty: int, lenx: int, leny: int):
    assert len(shape_mask) == 2
    mask = np.zeros(shape_mask, dtype=bool)
    mask[startx:startx + lenx, starty:starty + leny] = True
    return mask


def resize_batch(image_batch: NDArray, shape_image: Tuple) -> NDArray:
    images = []
    for image in image_batch:
        image_temp = cv2.resize(np.array(image).transpose(1, 2, 0),
                                dsize=shape_image).transpose(2, 0, 1)
        images.append(image_temp)
    images_np = np.stack(images)
    return images_np


class DiffusionImputer(Imputer):
    """
    UNDER DEVELOPMENT
    See https://github.com/andreas128/RePaint for details
    """
    def __init__(self, 
                 batch_size: int,
                 shape_input,
                 image_size: int,
                 device, 
                 config_file,
                 root_repaint: Path,
                 diffusion_steps: Any = None,
                 time_steps: Any = None,
                 n_resampling: Any = None,
                 class_conditioning: bool = False,
                 use_deepspeed: bool = False,
                 method_matching_internal_shape: str = 'empty_frame'):
        super().__init__()
        self.device = device
        self.match_shapes = method_matching_internal_shape

        self.batch_size = batch_size
        self.shape_input = shape_input
        self.external_image_size = image_size
        self.root_repaint = root_repaint

        # normalization for repaint
        self.normalize = Normalize(**imagenet_norm)
        self.unnormalize = UnNormalize(**imagenet_norm)

        # load config
        conf = conf_mgt.conf_base.Default_Conf()
        conf.update(yamlread(config_file))
        if diffusion_steps is not None:
             conf['diffusion_steps'] = diffusion_steps
        if time_steps is not None:
            conf['schedule_jump_params']['t_T'] = time_steps
        if n_resampling is not None:
            conf['schedule_jump_params']['jump_n_sample'] = n_resampling

        assert root_repaint.exists(), f'Please insert a valid RePaint root directory.\n{root_repaint}'

        conf.model_path = str(root_repaint / Path(conf.model_path))
        conf.classifier_path = str(root_repaint / Path(conf.classifier_path))
        # conf.image_size = 224
        # conf.data['eval']['lama_inet256_thin_n100_test']['image_size'] = 224
        
        # PredDiff theory: do not use class conditioning
        if class_conditioning:
            conf.class_cond = True
            self.cond_fn = create_cond_fn(conf, device) 
        else:
            conf.class_cond = False
            self.cond_fn = None
            #conf.classifier_scale = 0

        self.conf = conf
        # load model
        model, diffusion = create_model_and_diffusion(**select_args(conf, model_and_diffusion_defaults().keys()),
                                                      conf=conf)
        state_dict = torch.load(conf.model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        if conf.use_fp16:
            model.convert_to_fp16()
        model.eval()

        if use_deepspeed:
            # deepspeed
            import deepspeed
            from deepspeed.module_inject.containers import UNetPolicy
            # TODO, not working yet
            # init deepspeed inference engine
            deepspeed.init_inference(
                model=model,
                mp_size=1,        # Number of GPU
                dtype=torch.float16, # dtype of the weights (fp16)
                #replace_method="auto", # Lets DS autmatically identify the layer to replace
                replace_with_kernel_inject=False, # replace the model with the kernel injector-> does not work here
                #Dictionary mapping a client nn.Module to its corresponding injection policy. 
                #AKA injection_dict, provides the names of two linear layers as a tuple
                #Related issue: https://github.com/microsoft/DeepSpeed/issues/2602
                injection_policy={UNetModel: UNetPolicy} # this is just a guess, does not work. most likely this should contain tuples which map layers?
            )
            print("DeepSpeed Inference Engine initialized")

        self.model = model
        self.diffusion = diffusion
        self.imputer_name = 'diffusion'

    def numpy_to_tensor(self, x: NDArray) -> torch.Tensor:
        """
        Args: 
            x: shape (batch_size, height, width, n_channel)
        Returns:
            x: tensor on device, shape (batch_size, n_channel, height, width)
        """

        return torch.tensor(np.transpose(x, (0, 3, 1, 2)), device=self.device)

    @staticmethod
    def tensor_to_numpy(x: torch.Tensor) -> NDArray:
        """
        Args: 
            x: tensor on device, shape (batch_size, n_channel, height, width)
        Returns:
            x: shape (batch_size, height, width, n_channel)
        """
        x_np = x.cpu().detach().numpy()
        # return np.transpose(x_np, (0, 2, 3, 1))
        return x_np

    def model_fn(self, x, t, y=None, gt=None, **kwargs):
        assert y is not None
        #return self.model(x, t, y if self.conf.class_cond else None, gt=gt)
        return self.model(x, t, y, gt=gt)

    def _impute(self, data: NDArray, segmentation_coalitions: NDArray, n_imputations: int) \
            -> NDArray:
        """
        xxx
        Args:
            data: shape: (shape_x)
            segmentation_coalitions: shape: (n_masks, *shape_x)
            n_imputations: requested number of different imputations
        Returns:
            imputations: shape: (n_masks, *shape_input, *shape_x)
        """
        # _check_input(data, segmentation_coalitions)
        shape_output = (n_imputations, *segmentation_coalitions.shape)
        sample_fn = (self.diffusion.p_sample_loop if not self.conf.use_ddim else self.diffusion.ddim_sample_loop)
        shape_image_external = (3, self.external_image_size, self.external_image_size)
        # create masks from segmentation coalitions - here: keep=0, impute=1
        # True: keep, False: imputer
        mask_ = segmentation_coalitions > 0          # shape: (n_masks, *shape_input, *shape_x)

        # flatten input dimension
        mask = mask_.reshape((-1, *shape_image_external))
        n_masks = mask.shape[0]
        data = data.reshape(shape_image_external)

        masks = np.stack([mask for _ in range(n_imputations)])
        # shape: (n_imputations, n_masks, *shape_input, *shape_x)
        masks_flatten = 1. * masks.reshape((-1, *shape_image_external))  # loop over first dimension with batch_size

        data_stacked = np.stack([data for _ in range(len(masks_flatten))])

        # no need to normalize image -> already done during dataloading
        imputations = np.empty(shape=(n_imputations*n_masks, *shape_image_external))
        for idx in range(0, masks_flatten.shape[0], self.batch_size):
            # masks_batch = self.numpy_to_tensor(masks_flatten[idx:idx+self.batch_size])
            # data_batch = self.numpy_to_tensor(data_stacked[idx:idx+self.batch_size])
            masks_batch_external = torch.tensor(masks_flatten[idx:idx + self.batch_size])
            data_batch_external = torch.tensor(data_stacked[idx:idx + self.batch_size])
            batch_size = min(self.batch_size, len(data_batch_external))

            #if self.cond_fn is not None:
            y = torch.randint(low=0, high=NUM_CLASSES, size=(batch_size,), device=self.device)

            # pad data/mask with zeros to match (256, 256)
            shape_data = (batch_size, 3, self.conf.image_size, self.conf.image_size)
            if self.match_shapes == 'empty_frame':
                total_border_size = self.conf.image_size - self.external_image_size
                assert total_border_size >= 0, f'Image size is too large: {self.external_image_size} < {self.conf.image_size}'
                start = int(total_border_size / 2)
                mask_padding = make_center_mask(shape_mask=(self.conf.image_size, self.conf.image_size),
                                                startx=start, starty=start,
                                                lenx=self.external_image_size, leny=self.external_image_size)

                data_batch_internal = torch.zeros(shape_data, device=self.device, dtype=data_batch_external.dtype)
                masks_batch_internal = torch.zeros(shape_data, device=self.device, dtype=masks_batch_external.dtype)
                data_batch_internal[:, :, mask_padding] = data_batch_external.flatten(-2, -1)
                masks_batch_internal[:, :, mask_padding] = masks_batch_external.flatten(-2, -1)
            elif self.match_shapes == 'cv2_resize':
                shape_image_internal = (self.conf.image_size, self.conf.image_size)
                data_batch_internal = resize_batch(np.array(data_batch_external.detach().cpu()), shape_image=shape_image_internal)
                masks_batch_internal = resize_batch(np.array(masks_batch_external.detach().cpu()), shape_image=shape_image_internal)
                masks_batch_internal[masks_batch_internal > 0.5] = 1.
                masks_batch_internal[masks_batch_internal <= 0.5] = 0.

                imputations_external = resize_batch(data_batch_internal, shape_image=shape_image_external[1:])
            else:
                raise ValueError(f'Method not defined: method_matching_internal_shape != {self.match_shapes}.')

            # arr_gt = arr_gt.astype(np.float32) / 127.5 - 1

            masks_batch_internal_gpu = torch.tensor(masks_batch_internal, device=self.device,
                                                    dtype=torch.float32)
            data_batch_internal_gpu = torch.tensor(data_batch_internal, device=self.device,
                                                   dtype=torch.float32)

            assert masks_batch_internal_gpu.shape == shape_data
            assert data_batch_internal_gpu.shape == shape_data
            model_kwargs = {'gt': self.unnormalize(data_batch_internal_gpu) * 2. - 1.,
                            'gt_keep_mask': masks_batch_internal_gpu,
                            'y': y}
            result = sample_fn(self.model_fn,
                               shape=shape_data,
                               clip_denoised=self.conf.clip_denoised,
                               model_kwargs=model_kwargs,
                               cond_fn=self.cond_fn,
                               device=self.device,
                               progress=False,
                               return_all=True,
                               conf=self.conf)

            # normalize [-1, 1] back to [0, 1]
            imputations_internal = self.tensor_to_numpy(self.normalize((result['sample'] + 1.)/2.))
            # shape: (batch_size, 3, 256, 256)

            # print(f'imputations_internal.shape = {imputations_internal.shape}')
            # print(f'mask_padding.shape = {mask_padding.shape}')
            # print(f'shape_image_external = {shape_image_external}')
            # print(f'imputations[idx:idx+batch_size].shape = {imputations[idx:idx+batch_size].shape}')

            if self.match_shapes == 'empty_frame':
                imputations_external = imputations_internal[:, :, mask_padding].reshape((-1, *shape_image_external))
            elif self.match_shapes == 'cv2_resize':
                imputations_external = resize_batch(np.array(imputations_internal),
                                                    shape_image=shape_image_external[1:])
                # imputations_external = np.stack([resize_batch(np.array(single_imp_internal),
                #                                               shape_image=shape_image_external[1:])
                #                                  for single_imp_internal in imputations_internal])
            imputations[idx:idx+batch_size] = imputations_external    # shape:(batch_size, *shape_image_external)

        return imputations.reshape(shape_output)
