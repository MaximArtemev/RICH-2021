import torch
from omegaconf.dictconfig import DictConfig
from torchtyping import TensorType
from core.utils import DataTensorType, WeightTensorType, ContextTensorType
from typeguard import typechecked
from typing import Dict

import torch.optim as optim
from .model import RICHGAN


class Trainer:
    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.model = RICHGAN(config).to(config.utils.device)
        self.optim = {'G': optim.Adam(self.model.G.parameters(), lr=config.experiment.lr.G, betas=(0.5, 0.9)),
                      'C': optim.Adam(self.model.C.parameters(), lr=config.experiment.lr.C, betas=(0.5, 0.9))}
        self.names = {'G': self.model.G, 'C': self.model.C}

    @typechecked
    def train(self,
              module: str,
              data: DataTensorType,
              context: ContextTensorType,
              weight: WeightTensorType) -> float:
        self.optim[module].zero_grad()
        loss = self.model(module, data, context, weight, mode='train')
        loss.backward()
        if self.config.experiment.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.names[module].parameters(), self.config.experiment.grad_clip)
        self.optim[module].step()
        return loss.item()

    @typechecked
    def evaluate(self,
                 module: str,
                 data: DataTensorType,
                 context: ContextTensorType,
                 weight: WeightTensorType,
                 tag: str = 'training') -> Dict[str, float]:
        return self.model(module, data, context, weight, mode=tag)

    def save(self, path: str) -> None:
        states = {
            'G.model': self.model.G.state_dict(),
            'C.model': self.model.C.state_dict(),
            'G.optim': self.optim['G'].state_dict(),
            'C.optim': self.optim['C'].state_dict(),
        }
        torch.save(states, path)

    def load(self, path: str) -> None:
        states = torch.load(path, map_location=lambda storage, loc: storage)
        if 'G.model' in states:
            self.model.G.load_state_dict(states['G.model'])
        if 'C.model' in states:
            self.model.C.load_state_dict(states['C.model'])
        if 'G.optim' in states:
            self.optim['G'].load_state_dict(states['G.optim'])
        if 'C.optim' in states:
            self.optim['C'].load_state_dict(states['C.optim'])
