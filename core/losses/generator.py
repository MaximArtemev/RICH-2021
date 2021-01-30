import torch


def calculate_wgan_loss(critic, x_real, x_fake, context, weight):
    out_fake = critic(x_fake, context)
    return -torch.mean(out_fake * weight)


def calculate_jsgan_loss(critic, x_real, x_fake, context):
    pass

# todo add lsgan jsgan cramergan
