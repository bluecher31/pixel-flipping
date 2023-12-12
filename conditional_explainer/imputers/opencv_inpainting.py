import numpy as np
import cv2

from conditional_explainer.imputers.abstract_imputer import Imputer
from typing import Callable
from numpy.typing import NDArray

LOCAL = None


class OpenCVInpainting(Imputer):
    """
    This imputer is implemented within the popular SHAP framework.
    See https://docs.opencv.org/master/df/d3d/tutorial_py_inpainting.html for more details
    """
    def __init__(self,
                 normalize_image: Callable[[NDArray], NDArray] = None,
                 unnormalize_image: Callable[[NDArray], NDArray] = None,
                 inpainting_algorithm='telea'):
        """
        inpainting_algorithm: choices: ['telea', 'navier-stokes']
        """
        super(OpenCVInpainting, self).__init__()
        self.algorithm = inpainting_algorithm
        if self.algorithm == 'telea':
            self.inpaint_cv = cv2.INPAINT_TELEA
        elif self.algorithm == 'navier-stokes':
            self.inpaint_cv = cv2.INPAINT_NS
        else:
            assert False, f'Incorrect keyword inpainting_algorithm: {inpainting_algorithm}'
        self.unnormalize_image = unnormalize_image
        self.normalize_image = normalize_image

        self.imputer_name = f'cv2'

    def _impute(self, data: NDArray, segmentation_coalitions: NDArray, n_imputations: int) -> NDArray:
        """
          data: (..., 3, n_pixel, n_pixel), with 3 rgb channels


          Args:
              data: shape: (*shape_input, shape_x)
              segmentation_coalitions: shape: (n_masks, *shape_input, *shape_x)
              n_imputations: requested number of different imputations
          Returns:
              imputations: shape: (n_masks, *shape_input, *shape_x)
          """
        assert 3 == data.shape[-3], f'Expect rgb image with shape data.shape = (..., 3, width, height)'
        shape_x = (3, *data.shape[-2:])
        shape_input = data.shape[:-3]
        n_input = int(np.prod(shape_input))
        # n_samples = test_data.shape[0]
        # assert n_imputations == 1, 'only single shoot imputations allowed for this imputer'

        # imputations = np.zeros((1, *segmentation_coalitions.shape))

        imputations_list = []
        for i_image, image in enumerate(data.reshape((n_input, *shape_x))):
            segmentation_flattened = segmentation_coalitions.reshape((-1, n_input, *shape_x))
            for segmentation in segmentation_flattened[:, i_image]:
                # if bool(np.alltrue(segmentation < 0)) is True:      # impute the full image by mean rgb value
                if np.alltrue(segmentation < 0):
                    # original image when full image needs to be imputed
                    # if mask is not given then we mask the whole image
                    imputations_single = self._empty_image_baseline(image, segmentation, which='zero')

                else:
                    mask_impute = np.transpose(segmentation < 0, (1, 2, 0))
                    mask = mask_impute[..., 0]

                    img, rescale_factor = self._prepare_image(image)
                    dst = cv2.inpaint(src=img.astype(np.uint8), inpaintMask=mask.astype(np.uint8), inpaintRadius=3,
                                      flags=self.inpaint_cv)

                    imputations_single = np.transpose(dst, (2, 0, 1))/rescale_factor
                    if self.normalize_image is not None:
                        imputations_single = self.normalize_image(imputations_single)
                imputations_list.append(imputations_single)

        imputations_temp = np.stack(imputations_list).reshape(segmentation_coalitions.shape)
        imputations = np.stack([imputations_temp for _ in range(n_imputations)])

        return imputations

    def _empty_image_baseline(self, image: NDArray, segmentation: NDArray, which: str) -> NDArray:
        """Imputes the complete image."""
        if which == 'mean rgb':
            mean_rgb = image.mean((1, 2))
            imputations_single = mean_rgb[:, None, None] * np.ones_like(image)
        elif which == 'zero':
            imputations_single = np.zeros_like(image)
        elif which == 'gradient image':
            # select four random superpixel
            # get imputation
            # repeat with two other superpixels
            raise NotImplementedError
        else:
            raise ValueError(f'which = {which} is not defined. ')
        return imputations_single

    def _prepare_image(self, image: NDArray) -> [NDArray, int]:
        """Creates an int-valued RGB image."""
        if self.unnormalize_image is not None:  # unnormalize to floats
            image = self.unnormalize_image(image)

        img = np.transpose(image, (1, 2, 0))
        # rescale image to int value [0, 255]

        assert img.min() >= 0, 'please insert images in original format'
        if img.max() <= 1:  # value between 0 and 1
            rescale_factor = 255
        else:
            assert img.max() <= 255, 'invalid image format'
            rescale_factor = 1

        img *= rescale_factor
        return img, rescale_factor
