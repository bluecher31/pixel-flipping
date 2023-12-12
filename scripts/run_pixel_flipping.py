import torch            # needs to be imported at first position, otherwise torchvision is not loaded correctly

from src.experiments.attribution import load_attributions
from src.experiments.resources import load_model, load_imputer
from src.experiments.pixel_flipping import pixel_flipping, store_pixel_flipping

from src.experiments.attribution import root_attribution

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf
from pathlib import Path
import logging

from src.config.config import PixelFlippingConfig
from src.config.helpers import save_config, resolve_imagenet, load_all_configs

from tqdm import tqdm

cs = ConfigStore.instance()
cs.store(name="config", node=PixelFlippingConfig)
OmegaConf.register_new_resolver('resolve_imagenet', resolve_imagenet)


@hydra.main(config_path='../src/conf', config_name='run_pixel_flipping', version_base=None)
def main_pixel_flipping(cfg: PixelFlippingConfig):
    logging.info(msg='\n' + OmegaConf.to_yaml(cfg))
    save_config(cfg, Path.cwd())

    model = load_model(model_cfg=cfg.model, dataset_name=cfg.dataset.name)
    imputer = load_imputer(imputer_cfg=cfg.imputer_pf, dataset=cfg.dataset)

    root = root_attribution(cfg, root=Path(get_original_cwd()).parent / 'outputs')
    config_folders = load_all_configs(root=root) if cfg.explainer.name != 'random' else []
    images, attributions = load_attributions(cfg, config_folders)

    for image, attribution in tqdm(zip(images, attributions), total=len(images)):
        pf_result = pixel_flipping(cfg, image, attribution, imputer, model)
        store_pixel_flipping(pf_result)


if __name__ == '__main__':
    main_pixel_flipping()
