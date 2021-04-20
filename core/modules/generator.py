import torch
import torch.nn as nn
from torchtyping import TensorType
from typeguard import typechecked
from core.utils import DataTensorType, ContextTensorType, WeightTensorType
from omegaconf.dictconfig import DictConfig

from core.nn import LinearBlock
import logging

log = logging.getLogger(__name__)


class Generator(nn.Module):
    def __init__(self, config: DictConfig) -> None:
        super(Generator, self).__init__()
        self.config = config
        self.layers = self._build_layers()

    @typechecked()
    def _sample_latent(self, x: TensorType['batch', -1]):
        return torch.randn(x.shape[0], self.config.model.G.noise_dim, device=self.config.utils.device)

    def _build_layers(self) -> nn.ModuleList:
        layers = []

        for layer_index in range(self.config.model.G.num_layers):
            in_dim, out_dim = self.config.model.G.fc_dim, self.config.model.G.fc_dim
            if layer_index == 0: in_dim = self.config.model.G.noise_dim + self.config.experiment.data.context_dim
            if layer_index == self.config.model.G.num_layers - 1: out_dim = self.config.experiment.data.data_dim
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
                context: ContextTensorType) -> DataTensorType:
        noise = self._sample_latent(x)
        x = torch.cat([noise.float(), context.float()], dim=1)
        for layer in self.layers:
            x = layer(x)
        return x
