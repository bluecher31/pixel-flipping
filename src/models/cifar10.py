import torchvision

from src.interface import Model


def load_cifar10_model() -> Model:
    from torchvision.models import ResNet18_Weights
    model: Model
    print('hello cifar10 model')
    raise NotImplementedError
    return model