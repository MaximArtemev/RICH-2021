import torch
import torch.nn.functional as F
from torchtyping import TensorType
from typeguard import typechecked
from core.utils import DataTensorType, ContextTensorType, WeightTensorType
from core.modules import Critic


@typechecked()
def calculate_wgan_loss(critic: Critic,
                        x_real: DataTensorType,
                        x_fake: DataTensorType,
                        context: ContextTensorType,
                        weight: WeightTensorType) -> TensorType[()]:
    out_fake = critic(x_fake, context)
    return -torch.mean(out_fake * weight)


@typechecked()
def calculate_jsgan_loss(critic: Critic,
                        x_real: DataTensorType,
                        x_fake: DataTensorType,
                        context: ContextTensorType,
                        weight: WeightTensorType) -> TensorType[()]:
    out_fake = critic(x_fake, context)
    df_loss = F.binary_cross_entropy_with_logits(out_fake, torch.ones_like(out_fake), reduction='none')
    return torch.mean(df_loss * weight)


@typechecked()
def calculate_lsgan_loss(critic: Critic,
                        x_real: DataTensorType,
                        x_fake: DataTensorType,
                        context: ContextTensorType,
                        weight: WeightTensorType) -> TensorType[()]:
    out_fake = critic(x_fake, context) * weight
    df_loss = F.mse_loss(out_fake, torch.ones_like(out_fake), reduction='none')
    return torch.mean(df_loss * weight)

# todo add cramergan
