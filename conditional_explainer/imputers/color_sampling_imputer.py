import numpy as np

from conditional_explainer.imputers.abstract_imputer import Imputer
from numpy.typing import NDArray


class ColorHistogramImputer(Imputer):
    """
    Samples a uniform color according to the rgb distribution in the image which is estimated via b^3 histogram,
    Therefore, only applicable to rgb images.
    See https://ieeexplore.ieee.org/abstract/document/8546302 for details
    """
    def __init__(self, train_data: NDArray, n_bins=8):
        super().__init__()
        self.imputer_name = 'ColorHistogram'
        self.b = n_bins         # number of histogram bins per rbg channel
        self.colormin = train_data.min()
        self.colormax = train_data.max()
        self.splits = np.linspace(self.colormin, self.colormax, self.b)
        # self.hist_splits = np.broadcast_to(self.splits[:, np.newaxis], shape=(self.b, 3))
        self.rng = np.random.default_rng()

    def _impute(self, data: NDArray, segmentation_coalitions: NDArray, n_imputations: int):
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

        imputations_list = []
        for i_image, image in enumerate(data.reshape((n_input, *shape_x))):
            segmentation_flattened = segmentation_coalitions.reshape((-1, n_input, *shape_x))
            for segmentation in segmentation_flattened[:, i_image]:
                mask_impute = segmentation < 0
                # image = np.squeeze(test_data).transpose(1, 2, 0)
                image_flatten = image.reshape(3, -1).T
                H, edges = np.histogramdd(image_flatten, (self.splits, self.splits, self.splits))
                hist = H.flatten()
                colors_index = self.rng.choice(np.arange(hist.size), size=n_imputations, replace=True, p=hist / hist.sum())
                index_red, index_green, index_blue = np.unravel_index(colors_index, H.shape)

                imputations_image = np.stack([image.copy() for _ in range(n_imputations)])
                for i in range(n_imputations):
                    mask_red = np.logical_and(self.splits[index_red[i] + 1] > image_flatten[:, 0],
                                   image_flatten[:, 0] > self.splits[index_red[i]])
                    mask_green = np.logical_and(self.splits[index_green[i] + 1] > image_flatten[:, 1],
                                   image_flatten[:, 1] > self.splits[index_green[i]])
                    mask_blue = np.logical_and(self.splits[index_blue[i] + 1] > image_flatten[:, 2],
                                   image_flatten[:, 2] > self.splits[index_blue[i]])
                    mask = np.array([mask_red, mask_green, mask_blue]).all(axis=0)
                    color = image_flatten[mask].mean(axis=0)
                    color_broadcasted = np.broadcast_to(color[:, np.newaxis, np.newaxis], mask_impute.shape)
                    # TODO: discuss whether fixed imputed color is really desirable
                    imputations_image[i, mask_impute] = color_broadcasted[mask_impute]
                imputations_list.append(imputations_image)

        imputations_temp = np.swapaxes(np.stack(imputations_list), 0, 1)
        imputations = imputations_temp.reshape((n_imputations, *segmentation_coalitions.shape))

        return imputations
