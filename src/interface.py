import numpy as np
from dataclasses import dataclass

import torch

from typing import Protocol, List, Union, Dict, Optional
from numpy.typing import NDArray
from abc import ABC, abstractmethod


@dataclass
class Image:
    image: NDArray              # correct shape for predict_fn
    label: int                  # numeric class value
    image_name: str             # unique name for all (train, test, val, etc) images
    class_name: str             # meaningful expression for the target class
    dataset: str


@dataclass
class Attribution:
    heatmap: NDArray
    model: str
    prediction: NDArray
    image_name: str
    label: int
    dataset: str
    explainer: str
    explainer_properties: Dict[str, Union[str, int, float]]


@dataclass
class ConfigExplainer:
    name: str


class Model(Protocol):
    model_name: str

    def predict(self, images: Union[NDArray, List[Image]]) -> NDArray:
        """Predicts the calibrated probability of all classes."""


class ImageExplainer(ABC):
    explainer_name: str

    def __init__(self, model: Model, conf_explainer: ConfigExplainer):
        self.model = model
        self.conf = conf_explainer

    def attribute(self, images: List[Image]) -> List[Attribution]:
        """Calculates attributes for each pixel."""
        attributions = []
        for image in images:
            heatmap = self.get_heatmap(image=image)

            with torch.no_grad():
                model_prediction_image = self.model.predict(images=[image])
                model_prediction_image_np = np.array(model_prediction_image)

            attributions.append(Attribution(heatmap=heatmap,
                                            model=self.model.model_name,
                                            prediction=model_prediction_image_np,
                                            image_name=image.image_name,
                                            label=image.label,
                                            dataset=image.dataset,
                                            explainer=self.explainer_name,
                                            explainer_properties=dict(self.conf),
                                            ))
        return attributions

    @abstractmethod
    def get_heatmap(self, image: Image) -> NDArray:
        """Returns a pixel-wise heatmap for the given image."""


def convert_images_to_array(images: List[Image]) -> NDArray:
    """Extracts and stacks all images from list into an array with leading batch dimension."""
    return np.stack([image.image for image in images])


@dataclass
class Measurement:
    values: NDArray
    error: Optional[NDArray] = None

    def __post_init__(self):
        if self.error is not None:
            assert self.values.shape == self.error.shape, 'Non-compatible errors.'

    ...

    def __add__(self, other: 'Measurement') -> 'Measurement':
        values = self.values + other.values
        error = np.sqrt(self.error**2 + other.error**2)
        return Measurement(values, error)

    def __sub__(self, other: 'Measurement') -> 'Measurement':
        values = self.values - other.values
        error = np.sqrt(self.error**2 + other.error**2)
        return Measurement(values, error)

    def __rmul__(self, other: float) -> 'Measurement':
        values = other * self.values
        error = other * self.error
        return Measurement(values, error)
