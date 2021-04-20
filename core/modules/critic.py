import torch
import torch.nn as nn
from torchtyping import TensorType
from typeguard import typechecked
from core.utils import DataTensorType, ContextTensorType, WeightTensorType
from omegaconf.dictconfig import DictConfig

from core.nn import LinearBlock

import logging

log = logging.getLogger(__name__)


class Critic(nn.Module):
    def __init__(self, config: DictConfig) -> None:
        super(Critic, self).__init__()
        self.config = config
        self.layers = self._build_layers()

    def _build_layers(self) -> nn.ModuleList:
        layers = []

        for layer_index in range(self.config.model.C.num_layers):
            in_dim, out_dim = self.config.model.C.fc_dim, self.config.model.C.fc_dim
            if layer_index == 0: in_dim = self.config.experiment.data.data_dim + self.config.experiment.data.context_dim
            if layer_index == self.config.model.C.num_layers - 1: out_dim = 1
            layers.append(
                LinearBlock(
                    in_dim, out_dim,
                    normalization=self.config.model.G.normalization,
                    use_spectral=self.config.model.G.use_spectral
                )
            )
        layers[-1] = layers[-1].module
        return nn.ModuleList(layers)

    @typechecked()
    def forward(self,
                x: DataTensorType,
                context: ContextTensorType) -> TensorType['batch', -1]:
        x = torch.cat([x.float(), context.float()], dim=1)
        for layer in self.layers:
            x = layer(x)
        return x
