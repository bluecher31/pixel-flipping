import torch            # needs to be imported at first position, otherwise torchvision is not loaded correctly

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import logging

from pathlib import Path

from tqdm import tqdm

from src.config.helpers import save_config, resolve_imagenet
from src.experiments.attribution import save_attributions
from src.experiments.resources import load_model, load_explainer, load_data

from src.config.config import CalculateAttributions


cs = ConfigStore.instance()
cs.store(name="config", node=CalculateAttributions)
OmegaConf.register_new_resolver('resolve_imagenet', resolve_imagenet)


@hydra.main(config_path='../src/conf', config_name='calculate_attributions', version_base=None)
def main_calculate_attributions(cfg: CalculateAttributions):
    logging.info(msg='\n' + OmegaConf.to_yaml(cfg))
    save_config(cfg, Path.cwd())
    # load stuff
    test_images = load_data(dataset=cfg.dataset, n_samples=cfg.n_samples,
                            train=False, format_numpy=False)

    model = load_model(model_cfg=cfg.model, dataset_name=cfg.dataset.name)
    explainer = load_explainer(cfg_explainer=cfg.explainer, dataset=cfg.dataset, model=model)

    # calculate and store
    for image in tqdm(test_images):
        attributions = explainer.attribute(images=[image])

        save_attributions(attributions=attributions)


if __name__ == '__main__':
    main_calculate_attributions()
