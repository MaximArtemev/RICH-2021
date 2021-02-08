import torch
import logging

from core.modules import Generator, Critic
from .utils import LossWeighter
from core.losses import critic_wgan_loss, generator_wgan_loss, calculate_gradient_penalty

log = logging.getLogger(__name__)


class RICHGAN(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.lw = LossWeighter(config)

        self.G = Generator(config)
        self.C = Critic(config)

    # here we need forward for ddp to activate forward hook
    def forward(self, module, *args, **kwargs):
        if module == 'G':
            return self.trainG(*args, **kwargs)
        if module == 'C':
            return self.trainC(*args, **kwargs)

    def trainC(self, data, context, weight, mode='train'):
        losses = {}

        data_fake = self.G(data, context)
        losses['C.adversarial'] = critic_wgan_loss(self.C, data, data_fake, context, weight)
        if mode == 'train':
            losses['C.gradient_penalty'] = calculate_gradient_penalty(self.C, data, data_fake, context)

        if mode == 'train':
            return self.lw.combine_losses(losses)
        return self.lw.visualize_losses(losses, mode)

    def trainG(self, data, context, weight, mode='train'):
        losses = {}

        data_fake = self.G(data, context)
        losses['G.adversarial'] = generator_wgan_loss(self.C, data, data_fake, context, weight)

        if mode == 'train':
            return self.lw.combine_losses(losses)
        return self.lw.visualize_losses(losses, mode)

    @torch.no_grad()
    def generate(self, data, context):
        return self.G(data, context)
