import torch
import logging

from core.modules import Generator, Critic
from .utils import LossWeighter
from core.losses import calculate_gradient_penalty
from core.losses import critic_wgan_loss, critic_jsgan_loss, critic_lsgan_loss
from core.losses import generator_wgan_loss, generator_lsgan_loss, generator_jsgan_loss
from omegaconf.dictconfig import DictConfig
from core.utils import DataTensorType, WeightTensorType, ContextTensorType

from typeguard import typechecked

log = logging.getLogger(__name__)


class RICHGAN(torch.nn.Module):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.config = config
        self.lw = LossWeighter(config)

        self.G = Generator(config)
        self.C = Critic(config)

    # here we need forward for ddp to activate forward hook
    def forward(self, module: str, *args, **kwargs):
        if module == 'G':
            return self.trainG(*args, **kwargs)
        if module == 'C':
            return self.trainC(*args, **kwargs)

    @typechecked()
    def trainC(self,
               data: DataTensorType,
               context: ContextTensorType,
               weight: WeightTensorType,
               mode: str = 'train'):
        losses = {}

        data_fake = self.G(data, context).detach()

        if self.config.losses.adv_type == 'jsgan':
            losses['C.adversarial'] = critic_jsgan_loss(self.C, data, data_fake, context, weight)
        if self.config.losses.adv_type == 'lsgan':
            losses['C.adversarial'] = critic_lsgan_loss(self.C, data, data_fake, context, weight)
        if self.config.losses.adv_type == 'wgan':
            losses['C.adversarial'] = critic_wgan_loss(self.C, data, data_fake, context, weight)

        if mode == 'train' and self.config.losses.use_gp:
            losses['C.gradient_penalty'] = calculate_gradient_penalty(self.C, data, data_fake, context)

        if mode == 'train':
            return self.lw.combine_losses(losses)
        return self.lw.visualize_losses(losses, mode)

    @typechecked()
    def trainG(self,
               data: DataTensorType,
               context: ContextTensorType,
               weight: WeightTensorType,
               mode: str = 'train'):
        losses = {}

        data_fake = self.G(data, context)
        if self.config.losses.adv_type == 'jsgan':
            losses['G.adversarial'] = generator_jsgan_loss(self.C, data, data_fake, context, weight)
        if self.config.losses.adv_type == 'lsgan':
            losses['G.adversarial'] = generator_lsgan_loss(self.C, data, data_fake, context, weight)
        if self.config.losses.adv_type == 'wgan':
            losses['G.adversarial'] = generator_wgan_loss(self.C, data, data_fake, context, weight)

        if mode == 'train':
            return self.lw.combine_losses(losses)
        return self.lw.visualize_losses(losses, mode)

    @torch.no_grad()
    def generate(self,
                 data: DataTensorType,
                 context: ContextTensorType) -> DataTensorType:
        return self.G(data, context)
