import torch
import torch.nn.functional as F


def calculate_wgan_loss(critic, x_real, x_fake, context, weight):
    out_fake = critic(x_fake, context)
    return -torch.mean(out_fake * weight)


def calculate_jsgan_loss(critic, x_real, x_fake, context, weight):
    out_fake = critic(x_fake, context) * weight
    df_loss = F.binary_cross_entropy_with_logits(out_fake, torch.ones_like(out_fake))
    return df_loss

def calculate_lsgan_loss(critic, x_real, x_fake, context, weight):
    out_fake = critic(x_fake, context) * weight
    df_loss = F.mse_loss(out_fake, torch.ones_like(out_fake))
    return df_loss

# todo add cramergan
