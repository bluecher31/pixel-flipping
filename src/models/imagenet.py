# use pretrained ResNet-18 as default
# https://timm.fast.ai/
# https://pytorch.org/vision/master/models.html

import torch
from torch.nn import functional as F
import torchvision
import numpy as np

from src.interface import convert_images_to_array

from torch.utils.data import DataLoader, TensorDataset
from torchvision.io import read_image

from src.interface import Model, Image
from src.config.config import ModelConfig
from typing import Union, List, TypeVar, Tuple
from numpy.typing import NDArray


TNDArray = TypeVar('TNDArray', NDArray, torch.TensorType)

# img = read_image("test/assets/encode_jpeg/grace_hopper_517x606.jpg")
#
# # Step 1: Initialize model with the best available weights
# weights = ResNet50_Weights.DEFAULT
# model = resnet50(weights=weights)
# model.eval()

# # Step 2: Initialize the inference transforms
# preprocess = weights.transforms()
#
# # Step 3: Apply inference preprocessing transforms
# batch = preprocess(img).unsqueeze(0)
#
# # Step 4: Use the model and print the predicted category
# prediction = model(batch).squeeze(0).softmax(0)
# class_id = prediction.argmax().item()
# score = prediction[class_id].item()
# category_name = weights.meta["categories"][class_id]
# print(f"{category_name}: {100 * score:.1f}%")


def get_predict_proba_fn(model, batch_size: int, T: float, device: torch.device):
    def predict_proba(x: TNDArray) -> TNDArray:
        """If a torch.Tensor is provided we require gradients. """
        if isinstance(x, torch.Tensor):
            input_is_tensor = True
            x.requires_grad_()
        else:
            input_is_tensor = False
            x = torch.from_numpy(np.array(x)).float()
        test_dataset = TensorDataset(x)

        test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=False)

        if input_is_tensor:
            assert len(test_loader) == 1, 'To enable require_grad this code is restricted a a single batch forward pass.\n' \
                                          'Increase hparams.bs_test might be a possible option.'
        res = []
        model.eval()
        for data in test_loader:
            inputs = torch.tensor(data[0], device=device)
            # inputs.requires_grad_()
            preds = torch.nn.functional.softmax(model.forward(inputs) / T, dim=-1)
            # print(preds[0],preds.shape)
            res.append(preds)

        if input_is_tensor:
            # output = torch.concat(res, dim=0)
            output = res[0].detach()
        else:
            output = np.concatenate([r.detach().cpu() for r in res])

        return output
    return predict_proba


def downsample_to_patch_mask(segmentation: NDArray, patch_size: int, image_size: int = 224) \
        -> NDArray:
    """Create patch mask (internal dimensions of ViT tokens) from segmentation (image shape)."""
    assert segmentation.shape == (3, image_size, image_size), 'Not specified for other shapes. '

    shape_patches = (patch_size, patch_size)
    patch_mask = np.zeros(shape_patches, dtype=int)
    coarse_grain_factor, remainder = divmod(image_size, patch_size)
    assert remainder == 0, f'Image size and patch_size do not match: \n' \
                           f'remainder={image_size % patch_size} =! 0'

    mask_input = segmentation > 0

    for row_patch, row_in in enumerate(range(0, image_size, coarse_grain_factor)):
        for col_patch, col_in in enumerate(range(0, image_size, coarse_grain_factor)):
            current_window = mask_input[:, row_in:row_in + coarse_grain_factor, col_in:col_in + coarse_grain_factor]
            assert np.alltrue(current_window) or np.alltrue(~current_window), \
                f'Could not match segmentation mask to internal patch mask, \n' \
                f'which are squares of {coarse_grain_factor, coarse_grain_factor} in input space.'
            keep_patch = np.alltrue(current_window)  # This patch is not imputed!
            patch_mask[row_patch, col_patch] = keep_patch
    return patch_mask


def load_imagenet_model(cfg_model: ModelConfig) -> Model:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from torchvision.models import resnet50, resnet18
    if cfg_model.name == 'ResNet50':
        # weights = ResNet50_Weights.DEFAULT
        model = resnet50(pretrained=True)
    elif cfg_model.name == 'ResNet18':
        # weights = ResNet18_Weights.DEFAULT
        model = resnet18(pretrained=True)
    elif cfg_model.name.split('_')[0] == 'timm':
        import timm
        arch = cfg_model.name[5:]
        model = timm.create_model(arch.lower(), pretrained=True)
    elif cfg_model.name == 'Madry_ViT':
        import src.models.vision_transformer_patched as vitp
        assert cfg_model.image_size == 224 and cfg_model.patch_size == 14, \
            'Image size and patch_size cannot be altered.'
        model = vitp.deit_small_patch16_224(pretrained=True)

        def occluded_model(data: NDArray, segmentation: NDArray) -> NDArray:
            """
            Returns occluded prediction rescaled by the specified transformation (e.g. identity or log2).

            Notes:
                This function can potentially be overwritten externally.
            Args:
                data: (*shape_input, *shape_x), usually shape_input = (n_samples) but more dimensions are possible
                segmentation: (n_int, n_coalitions, *shape_input, *shape_x)
            Returns:
                occluded predictions, shape: (n_int, n_coalitions, shape_input, n_classes)
            """
            assert data.shape == segmentation.shape

            patch_mask = downsample_to_patch_mask(segmentation, image_size=cfg_model.image_size,
                                                  patch_size=cfg_model.patch_size)
            # their code also supports batch-wise masks (masking value -1) we use a global mask here
            # patch_mask = torch.ones(14 * 14 + 1, dtype=torch.bool)
            # plt.imshow(patch_mask[1:].numpy().reshape(14, 14))
            tensor = torch.tensor(data, device=device)
            patch_mask_tensor = torch.tensor(patch_mask, device=device)
            flat_patch_mask = torch.cat([torch.tensor([True], device=device), patch_mask_tensor.flatten()])
            # out = model(tensor[None], torch.ones(14**2+1, dtype=torch.bool))
            out = model(tensor[None], torch.tensor(flat_patch_mask, dtype=torch.bool, device=device))
            probabilities = torch.nn.functional.softmax(out, dim=-1)
            # print(probabilities.max())
            return probabilities.detach().cpu().numpy()

        model.occluded_model_fn = occluded_model
        # config = resolve_data_config({}, model=model)
        # config[
        #     "interpolation"] = "bicubic"  # using the same value as the native model from timm- crop_pct differs for some weird reason (we should check if )
        # transform = create_transform(**config)

    else:
        raise ValueError(f'This imagenet model is not defined: {cfg_model.name}')
    model.eval()

    # Step 2: Initialize the inference transforms
    # preprocess = weights.transforms()

    model.to(device)

    model.predict_proba = get_predict_proba_fn(model=model, batch_size=128, T=1., device=device)

    def predict(images: Union[TNDArray, List[Image]]) -> TNDArray:
        if isinstance(images, list):
            samples = convert_images_to_array(images)
        elif isinstance(images, torch.Tensor):
            samples = images
        else:
            samples = np.array(images)
        return model.predict_proba(samples)

    model.predict = predict
    model.model_name = cfg_model.name

    return model
