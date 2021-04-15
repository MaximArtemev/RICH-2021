import numpy as np
from omegaconf.dictconfig import DictConfig
from torchtyping import TensorType
from typing import Dict, Any, Tuple
from typeguard import typechecked
from torch.utils.data import DataLoader

DataTensorType = TensorType['batch', 'dll': 5]
ContextTensorType = TensorType['batch', 'context': 3]
WeightTensorType = TensorType['batch', 'weight': 1]
DataContextTensorType =  TensorType['batch', 'dll_context': 8]

class LossWeighter:
    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.weights = DotDict(config.losses)

    @typechecked
    def combine_losses(self, pairs: Dict[str, TensorType[()]]):
        combined_loss = 0
        for loss_name, loss in pairs.items():
            if self.weights._getattr(loss_name) != 0:
                combined_loss += self.weights._getattr(loss_name) * loss
        return combined_loss

    def visualize_losses(self, pairs: Dict[str, TensorType[()]], mode: str):
        visualized_losses = {}
        for loss_name, loss in pairs.items():
            text = f"{mode}/{'/'.join(loss_name.split('.'))}"
            visualized_losses[text] = loss.item()
        return visualized_losses


class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct: Dict[str, Any]) -> None:
        super().__init__()
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value

    def _getattr(self, key: str) -> Any:
        target = self
        for dot in key.split('.'):
            target = target[dot]
        return target

    def _setattr(self, key: str, value: Any) -> None:
        target = self
        for dot in key.split('.')[:-1]:
            target = target[dot]
        target[key.split('.')[-1]] = value


class InfiniteDataloader:
    def __init__(self, loader: DataLoader) -> None:
        self.loader = loader
        self.iter = iter(self.loader)

    def get_next(self) -> Tuple[DataTensorType, ContextTensorType, WeightTensorType]:
        try:
            return next(self.iter)
        except StopIteration:
            self.iter = iter(self.loader)
            return self.get_next()
