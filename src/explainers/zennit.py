import torch
import numpy as np

import zennit
from zennit.composites import EpsilonPlusFlat
from zennit.attribution import Gradient

from pathlib import Path
from dataclasses import dataclass

from src.interface import Image, ImageExplainer, Model, ConfigExplainer
from numpy.typing import NDArray


@dataclass
class ConfigZennit(ConfigExplainer):
    composite_name: str


@dataclass
class ConfigGradient(ConfigExplainer):
    dummy: str = 'gradient'


class ZennitGradient(ImageExplainer):
    def __init__(self, model: Model, conf_explainer: ConfigGradient):
        self.explainer_name = 'Gradient'
        super().__init__(model=model, conf_explainer=conf_explainer)

    def get_heatmap(self, image: Image) -> NDArray:
        input = torch.tensor(image.image[None])
        input.requires_grad_()
        n_classes = 10
        target = torch.eye(n_classes)[[image.label]]

        with Gradient(model=self.model) as attributor:
            # compute the model output and attribution
            output, attribution = attributor(input, target)
        heatmap_np = np.squeeze(attribution.detach().numpy())
        return heatmap_np


class ZennitExplainer(ImageExplainer):
    def __init__(self, model: Model, conf_explainer: ConfigZennit):
        self.explainer_name = 'LRP'
        super().__init__(model=model, conf_explainer=conf_explainer)
        if conf_explainer.composite_name == 'EpsilonPlusFlat':
            self.composite = EpsilonPlusFlat()
        else:
            raise NotImplementedError(f'This composite is not implemented yet: {conf_explainer.composite_name}.')

        self.attributor = Gradient(self.model, self.composite)

        # self.ig = attr.IntegratedGradients(self.model.predict)

    def get_heatmap(self, image: Image) -> NDArray:
        input = torch.tensor(image.image[None])
        input.requires_grad_()
        if image.dataset == 'mnist':
            n_classes = 10
        elif image.dataset == 'imagenet':
            n_classes = 1000
        else:
            raise ValueError(f'How many classes does this dataset ({image.dataset}) have?')

        target = torch.eye(n_classes)[[image.label]]

        output_orginal = self.model(torch.tensor(image.image[None]))

        with self.attributor as attributor:
            output1, relevance = attributor(input, target)
        with self.composite.context(self.model) as modified_model:
            output2 = modified_model(input)
            # gradient/ relevance wrt. class/output 0
            output2.backward(gradient=target)
            # relevance is not accumulated in .grad if using torch.autograd.grad
            # relevance, = torch.autograd.grad(output, input, torch.eye(10)[[0])
            heatmap = input.grad

        with Gradient(model=self.model) as attributor:
            # compute the model output and attribution
            output, attribution = attributor(input, target)
        heatmap_np = np.squeeze(heatmap.detach().numpy())
        return heatmap_np
