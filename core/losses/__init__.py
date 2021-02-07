from .critic import calculate_gradient_penalty
from .critic import calculate_wgan_loss as critic_wgan_loss
from .critic import calculate_jsgan_loss as critic_jsgan_loss
from .critic import calculate_lsgan_loss as critic_lsgan_loss

from .generator import calculate_wgan_loss as generator_wgan_loss
from .generator import calculate_jsgan_loss as generator_jsgan_loss
from .generator import calculate_lsgan_loss as generator_lsgan_loss