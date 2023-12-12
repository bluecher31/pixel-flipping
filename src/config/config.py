from dataclasses import dataclass

from src.interface import ConfigExplainer

from typing import List


@dataclass
class DatasetConfig:
    name: str
    root_data: str
    image_size: int


@dataclass
class ModelConfig:
    name: str


@dataclass
class ImputerConfig:
    name: str


@dataclass
class SegmentationConfig:
    name: str
    n_superpixel: int


@dataclass
class ImputationsConfig:
    segmentation: SegmentationConfig
    imputer: ImputerConfig
    dataset: DatasetConfig

    n_samples: int


# TODO: unify and factorize with imputation Config
@dataclass
class MeasureOODScoresConfig:
    dataset: DatasetConfig
    segmentation: SegmentationConfig
    imputer: ImputerConfig
    model: ModelConfig
    n_samples: int


@dataclass
class VisualizeOODScoresConfig:
    dataset: DatasetConfig
    segmentation: SegmentationConfig
    imputer_names: List[str]
    model: ModelConfig
    n_samples: int


@dataclass
class CalculateAttributions:
    dataset: DatasetConfig
    explainer: ConfigExplainer
    model: ModelConfig
    n_samples: int


@dataclass
class PixelFlippingConfig:
    imputer_pf: ImputerConfig
    segmentation_pf: SegmentationConfig
    dataset: DatasetConfig
    model: ModelConfig
    explainer: ConfigExplainer
    n_samples: int

    minimal_percentage: float
    maximal_percentage: float
    n_measurements: int
    most_relevant_first: bool
    n_imputations_pf: int
