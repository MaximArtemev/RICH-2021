import torch
import torch.nn.functional as F


def interpolate(a, b):
    alpha = torch.rand(a.size(0), 1, device=a.device)
    inter = a + alpha * (b - a)
    return inter


def calculate_gradient_penalty(critic, x_real, x_fake, context):
    image = interpolate(x_real.detach(),
                        x_fake.detach()).requires_grad_(True)
    pred = critic(image, context)
    grad = torch.autograd.grad(
        outputs=pred, inputs=image,
        grad_outputs=torch.ones_like(pred),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad = grad.view(grad.shape[0], -1)
    norm = grad.norm(2, dim=1)
    gp = ((norm - 1.0) ** 2).mean()
    return gp


def calculate_wgan_loss(critic, x_real, x_fake, context, weight):
    out_real = critic(x_real, context) * weight # dont like it
    out_fake = critic(x_fake, context) * weight # dont like it
    return torch.mean(out_fake) - torch.mean(out_real)


def calculate_jsgan_loss(critic, x_real, x_fake, context, weight):
    out_real = critic(x_real, context)
    out_fake = critic(x_fake, context)
    df_loss = F.binary_cross_entropy_with_logits(out_real, torch.ones_like(out_real), reduction='none') + \
              F.binary_cross_entropy_with_logits(out_fake, torch.zeros_like(out_fake), reduction='none')
    return torch.mean(df_loss * weight)


def calculate_lsgan_loss(critic, x_real, x_fake, context, weight):
    out_real = critic(x_real, context)
    out_fake = critic(x_fake, context)
    df_loss = F.mse_loss(out_real, torch.ones_like(out_real), reduction='none') + \
              F.mse_loss(out_fake, torch.zeros_like(out_fake), reduction='none')
    return torch.mean(df_loss * weight)

# todo add cramergan
