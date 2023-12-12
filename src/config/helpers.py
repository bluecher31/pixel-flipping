import copy
from datetime import datetime

from pathlib import Path
from functools import partial
from typing import List, Callable, Union, Tuple, Any

import omegaconf.errors
from omegaconf import OmegaConf, DictConfig
from hydra.utils import get_original_cwd


def resolve_imagenet(platform: str) -> str:
    """Returns path to imagenet folder."""
    # TODO: maybe load these file_path from an external yaml file?
    if platform == 'local_stefan':
        path = Path(get_original_cwd()).parent.parent / 'data/ImageNet-complete/val'
        root_data = str(path)
    elif platform == 'ida_cluster':
        root_data = '/home/space/datasets/imagenet/2012/val_set'
    elif platform == 'local_johanna':
        root_data = '/media/vielhaben/Data1/ImageNet-complete/val'
    else:
        raise ValueError(f'args.platform = {platform}')
    return root_data


def map_name_to_yaml_file(name: str) -> str:
    """Mapping the internal name of objects to the corresponding yaml config file name. default: pass identity."""
    if name == 'ConstantValueImputer':
        return 'constant_value'
    elif name == 'ColorHistogram':
        return 'color_histogram'
    # elif name == 'TrainSet':
    #     return 'trainset'
    # elif name == 'ColorHistogram':
    #     return 'color_histogram'
    else:
        return name


def compare_cfg_default(cfg1: DictConfig, cfg2: DictConfig):
    """Comparing the full config."""
    return cfg1 == cfg2


def compare_cfg(cfg_base: DictConfig, cfg_test: DictConfig, keys: List[str], set_keys: List[str] = []) -> bool:
    """
    Custom comparison between two DictConfigs.

    Args:
        keys: These keys are explicitly compared
        set_keys: the values of cfg_test are set to the values of cfg_base
    """
    for set_key in set_keys:
        if 'dataset.root_data' == set_key:
            cfg_test['dataset']['root_data'] = cfg_base['dataset']['root_data']
        else:
            cfg_test[set_key] = cfg_base.get(set_key)

    for key in keys:
        if cfg_base.get(key) != cfg_test.get(key):
            # print(f'cfg_base[{key}] != cfg_test[{key}]\n'
            #       f'{cfg_base[key]} != {cfg_test[key]}')
            return False
    return True


def load_all_configs(root: Path) -> List[Tuple[Path, DictConfig]]:
    assert root.exists(), f'Could not find folder associated with the params.\n' \
                          f'{root}\n'

    config_folders = []
    for folder in root.glob('*/*'):
        path_to_config = folder / 'config.yaml'
        if path_to_config.exists():
            cfg = OmegaConf.load(path_to_config)
            config_folders.append((folder, cfg))
    assert len(config_folders) > 0, f'No data folders in root. \n{root}'
    return config_folders


def load_special_config(name: str, type: str) -> DictConfig:
    """type in ['imputer', 'segmentation', 'explainer']"""
    name_internal = map_name_to_yaml_file(name)
    if type in ['imputer', 'segmentation', 'explainer', 'model']:
        path_to_config = Path(get_original_cwd()).parent / f'src/conf/{type}/{name_internal}.yaml'
    else:
        raise ValueError(f'type={type} is not defined.')
    return OmegaConf.load(path_to_config)


def get_value(cfg: DictConfig, key: str) -> Any:
    """
    Extract the value associated with key from cfg.

    key: 'key1.key2' -> cfg.get(key1).get(key2)

    """
    cfg_temp = copy.deepcopy(cfg)
    for unique_key in key.split('.'):
        cfg_temp = cfg_temp.get(unique_key)
    return cfg_temp


def filter_config_folders(config_folders: List[Tuple[Path, DictConfig]], key: str, value: Any) \
        -> List[Tuple[Path, DictConfig]]:
    config_folders_filtered = []
    for (path, cfg_test) in config_folders:
        value_test = get_value(cfg_test, key)
        if value_test == value:
            config_folders_filtered.append((path, cfg_test))
    return config_folders_filtered


def save_config(cfg: DictConfig, output_dir: Path):
    output_dir.mkdir(exist_ok=True, parents=True)
    assert output_dir.is_dir()
    with open(str(output_dir / 'config.yaml'), "w", encoding="utf-8") as file:
        file.write(OmegaConf.to_yaml(cfg))


def find_all_existing_path(folder: Path, file_path: [Path, str], pattern='*') -> List[Path]:
    """Find all folder.glob(pattern)/file_path and checks for existence. Returns sorted list."""
    all_folders = [path / file_path for path in folder.glob(pattern=pattern)]
    all_existing = [path for path in all_folders if path.exists()]
    return sorted(all_existing)


def get_datetime(folder: Path) -> datetime:
    date = folder.parent.name
    time = folder.name
    datetime_str = f'{date}_{time}'
    if len(datetime_str) == 26:
        format = '%Y-%m-%d_%H-%M-%S.%f'
    else:
        format = '%Y-%m-%d_%H-%M-%S'
    return datetime.strptime(datetime_str, format)


def select_folder(folders: List[Path], rule: Union[int, str], index: int = 0) -> Path:
    """
    Allows to select a single folder based on some rules, i.e. index or datetime.

    rule:
        int: select the corresponding folder = folder[int]
        'new': choose the newest folder in term of datetime
        'most_subfolders': count the number of subfolders as a proxy for the number of measurements
    index: sort according to rule and then apply 'index' offset

    """
    assert len(folders) > 0, 'Please enter a least a single folder.'
    if isinstance(rule, int):
        return folders[rule]

    elif rule == 'new':
        folders.sort()
        folders_sorted_by_datetime = sorted(folders, key=get_datetime)
        return folders_sorted_by_datetime[-(1 + index)]

    elif rule == 'most_subfolders':
        def get_number_of_subfolders(folder: Path) -> int:
            subfolders = [f for f in folder.glob('*') if f.is_dir() and not f.name.startswith('.')]
            return len(subfolders)
        folders.sort()
        folders = sorted(folders, key=get_datetime)
        folders_sorted_by_subfolders = sorted(folders, key=get_number_of_subfolders)
        return folders_sorted_by_subfolders[-(1 + index)]

    else:
        raise ValueError(f'Please insert a valid rule: {rule}')


def _find_data_folders(base_dir: Path) -> [List[Path], List[Path]]:
    """Return all datafolders and remaining searchable paths."""
    undefined_subfolders = []
    data_folders = []
    assert base_dir.is_dir()
    for path in base_dir.glob('*'):
        full_content = [data_path for data_path in path.glob('*')]
        if (path / 'config.yaml').exists():         # found a data folder
            data_folders.append(path)
        else:           # ordinary folder
            if len(full_content) == 0:      # emtpy folder
                pass
            else:
                undefined_subfolders.append(path)

    return data_folders, undefined_subfolders


def find_all_data_dirs(base_folder: Path) -> List[Path]:
    """Remove all folders without data."""
    assert base_folder.exists(), f'Please enter a valid base folder.\n{base_folder}'
    max_depth = 20

    all_empty_data_folders = []
    current_undefined_folders = [base_folder]
    for _ in range(max_depth):
        remaining_undefined_subfolders = []
        for folder in current_undefined_folders:
            empty_data_folders, undefined_subfolders = _find_data_folders(base_dir=folder)
            remaining_undefined_subfolders += undefined_subfolders
            all_empty_data_folders += empty_data_folders
        if len(remaining_undefined_subfolders) == 0:          # nothing more to search
            break
        else:
            current_undefined_folders = remaining_undefined_subfolders
    return sorted(all_empty_data_folders)


def calculate_n_samples(path: Path) -> int:
    full_content = [data_path for data_path in path.glob('*')]
    n_samples = len([p for p in full_content if p.is_dir() or p.suffix == '.pickle']) - 1  # remove .hydra
    return n_samples


def check_for_nsamples(path: Path, n) -> bool:
    return calculate_n_samples(path) <= n


def eval_operator(value: [int, float], operation: [int, float, str]) -> bool:
    if isinstance(operation, int) or isinstance(operation, float):
        check = value == operation
    elif isinstance(operation, str):
        evaluation_str = f'{value} {operation}'
        check = eval(evaluation_str)
        assert isinstance(check, bool), f'Expecting a boolean expression to be evaluated: {evaluation_str}.'
    else:
        raise NotImplementedError
    return check


def get_check_kwargs(**kwargs) -> Callable[[Path], bool]:
    def check_fn(path: Path) -> bool:
        # full_content = [data_path for data_path in path.glob('*')]
        cfg = OmegaConf.load(path / 'config.yaml')
        # check = False       # default: config does match
        try:
            for key in kwargs:
                if key == 'type':       # check for correct experiment
                    current_check = kwargs['type'] in path.parts
                elif key == 'dataset':
                    current_check = cfg['dataset']['name'] == kwargs[key]
                elif key == 'n_samples':
                    if key in cfg:
                        current_check = eval_operator(value=cfg[key], operation=kwargs[key])
                    else:
                        current_check = False
                elif key == 'n_eval':
                    if 'explainer' in cfg:
                        current_check = eval_operator(value=cfg['explainer']['n_eval'], operation=kwargs[key])
                    else:
                        current_check = False
                elif key == 'cardinality_coalitions':
                    if 'explainer' in cfg:
                        current_check = eval_operator(value=cfg['explainer']['cardinality_coalitions'],
                                                      operation=kwargs[key])
                    else:
                        current_check = False
                elif key == 'model':
                    if 'model' in cfg:
                        current_check = cfg['model']['name'] == kwargs[key]
                    else:
                        current_check = False
                elif key == 'explainer':
                    if 'explainer' in cfg:
                        current_check = cfg['explainer']['name'] == kwargs[key]
                    else:
                        current_check = False
                elif key == 'imputer':
                    if 'explainer' in cfg:
                        current_check = cfg['explainer']['imputer']['name'] == kwargs[key]
                    elif 'imputer' in cfg:
                        current_check = cfg['imputer']['name'] == kwargs[key]
                    else:
                        current_check = False
                elif key == 'imputer_pf':
                    if 'imputer_pf' in cfg:
                        current_check = cfg['imputer_pf']['name'] == kwargs[key]
                    else:
                        current_check = False
                elif key == 'n_superpixel':
                    if 'explainer' in cfg:
                        current_check = eval_operator(value=cfg['explainer']['segmentation']['n_superpixel'],
                                                      operation=kwargs[key])
                    elif 'segmentation' in cfg:
                        current_check = eval_operator(value=cfg['segmentation']['n_superpixel'],
                                                      operation=kwargs[key])
                    else:
                        current_check = False
                else:
                    raise ValueError(f'key: {key} in kwargs is not defined.')

                if current_check is False:
                    return False
        except omegaconf.errors.ConfigKeyError:     # if key does not exist -> cfg does not match
            return False

        return True     # default
    return check_fn


def filter_data_dirs(base_folder: Path, check_fn: Callable[[Path], bool]) -> List[Path]:
    data_dirs = find_all_data_dirs(base_folder)
    empty_data_dirs = [folder for folder in data_dirs if check_fn(folder)]
    return empty_data_dirs


def remove_dirs(paths: List[Path]):
    import shutil
    print(f'Deleting {len(paths)} directories...')
    for path in paths:
        assert path.is_dir()
        shutil.rmtree(path)
    print('All directories are successfully deleted.')


def print_information(paths: List[Path], extended: bool = False, surpress_empty_dir: bool = False):
    if len(paths) == 0:
        print('Found nothing.')

    for path in paths:
        n_data_folders = calculate_n_samples(path)
        if n_data_folders == 0 and surpress_empty_dir is True:
            next
        else:
            cfg = OmegaConf.load(path / 'config.yaml')
            if extended:
                print(f'n={calculate_n_samples(path)} - {path}')
            print(cfg)


if __name__ == '__main__':
    delete_emtpy = True

    base_folder = Path('.').absolute().parent.parent / 'outputs'
    if delete_emtpy is True:
        check_for_empty_dir = partial(check_for_nsamples, n=0)
        empty_dirs = filter_data_dirs(base_folder=base_folder, check_fn=check_for_empty_dir)
        remove_dirs(paths=empty_dirs)

    data_dirs = find_all_data_dirs(base_folder)
    kwargs = {
        'type': 'pixel_flipping',
        'dataset': 'imagenet',
        'model': 'timm_resnet50',
        # 'n_eval': '==1000',
        'n_samples': '==100',
        # 'cardinality_coalitions': '== [-1]',
        # 'explainer': 'IntegratedGradients',          # name
        # 'imputer': 'diffusion',
        # 'imputer_pf': 'diffusion',
        'n_superpixel': '==25'
    }

    remaining_data_dirs = filter_data_dirs(base_folder, check_fn=get_check_kwargs(**kwargs))
    # remove_dirs(remaining_data_dirs)
    print_information(remaining_data_dirs, extended=True, surpress_empty_dir=True)

