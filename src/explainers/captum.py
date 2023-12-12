import torch
import numpy as np
from dataclasses import dataclass

from captum import attr

from src.explainers.helper_images import generate_superpixel

from src.interface import Image, ImageExplainer, Model, ConfigExplainer
from numpy.typing import NDArray


@dataclass
class ConfigIntegratedGradients(ConfigExplainer):
    n_steps: int
    baseline: float


@dataclass
class ConfigCaptumShapley(ConfigExplainer):
    baseline: float
    n_superpixel: int
    compactness_slic: float
    n_eval: int


class Gradients(ImageExplainer):
    def __init__(self, model: Model, conf_explainer: ConfigIntegratedGradients):
        self.explainer_name = 'Gradients'
        super().__init__(model=model, conf_explainer=conf_explainer)
        self.explainer = attr.Saliency(self.model)
        if conf_explainer.noise_tunnel is True:
            self.explainer = attr.NoiseTunnel(self.explainer)

    def get_heatmap(self, image: Image) -> NDArray:
        image_tensor = torch.tensor(image.image[None], requires_grad=True)

        heatmap = self.explainer.attribute(image_tensor, target=image.label)
        heatmap_np = np.squeeze(heatmap.detach().numpy())
        return heatmap_np


class InputXGradients(ImageExplainer):
    def __init__(self, model: Model, conf_explainer: ConfigIntegratedGradients):
        self.explainer_name = 'Gradients'
        super().__init__(model=model, conf_explainer=conf_explainer)
        self.gradient = attr.InputXGradient(self.model)

    def get_heatmap(self, image: Image) -> NDArray:
        image_tensor = torch.tensor(image.image[None], requires_grad=True)

        heatmap = self.gradient.attribute(image_tensor, target=image.label)
        heatmap_np = np.squeeze(heatmap.detach().numpy())
        if self.conf.abs is True:
            heatmap_np = np.abs(heatmap_np)
        return heatmap_np


class IntegratedGradients(ImageExplainer):
    def __init__(self, model: Model, conf_explainer: ConfigIntegratedGradients):
        self.explainer_name = 'IntegratedGradients'
        super().__init__(model=model, conf_explainer=conf_explainer)
        self.explainer = attr.IntegratedGradients(self.model)
        if conf_explainer.noise_tunnel is True:
            self.explainer = attr.NoiseTunnel(self.explainer)

    def get_heatmap(self, image: Image) -> NDArray:
        image_tensor = torch.tensor(image.image[None], requires_grad=True)

        heatmap = self.explainer.attribute(image_tensor,
                                           # baselines=self.conf.baseline,
                                           method='gausslegendre',
                                           n_steps=self.conf.n_steps,
                                           return_convergence_delta=False, target=image.label)
        heatmap_np = np.squeeze(heatmap.detach().numpy())
        if self.conf.abs is True:
            heatmap_np = np.abs(heatmap_np)
        return heatmap_np


class CaptumShapleyValues(ImageExplainer):
    def __init__(self, model: Model, conf_explainer: ConfigCaptumShapley):
        self.explainer_name = 'CaptumShapleyValues'
        super().__init__(model=model, conf_explainer=conf_explainer)
        self.explainer = attr.ShapleyValueSampling(self.model.predict)

    def get_heatmap(self, image: Image) -> NDArray:
        segmentation = generate_superpixel(image, self.conf.n_superpixel)
        heatmap = self.explainer.attribute(inputs=torch.tensor(image.image[None]),
                                           baselines=self.conf.baseline,
                                           target=image.label,
                                           feature_mask=torch.tensor(segmentation[None] - 1),  # min: 0
                                           n_samples=self.conf.n_eval)
        heatmap_np = np.squeeze(heatmap.detach().numpy())
        return heatmap_np
