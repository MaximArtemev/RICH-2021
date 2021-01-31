import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.distributed as dist
import wandb

from core.data import DataHandler
from core.utils import InfiniteDataloader
from core.trainer import Trainer
# from core.metrics import calculate_fid
# todo add fid
import logging
from omegaconf import OmegaConf

log = logging.getLogger(__name__)


def setup_experiment(config):
    wandb.login()
    wandb.init(project=config.wandb.project_name,
               config=OmegaConf.to_container(config, resolve=True),
               mode='online' if config.wandb.enabled else 'disabled')
    wandb.config.run_dir = os.getcwd()
    save_path = os.path.join(os.getcwd(), '.hydra', 'config.yaml')
    wandb.save(save_path)


def train(gpu_num_if_use_ddp, config):
    if config.utils.use_ddp:
        dist.init_process_group(
            backend="nccl",
            init_method="tcp://127.0.0.1:4433",
            world_size=torch.cuda.device_count(),
            rank=gpu_num_if_use_ddp,
        )
        config.utils.device = f'cuda:{gpu_num_if_use_ddp}'
        main_node = gpu_num_if_use_ddp == 0
    else:
        main_node = True
    if main_node:
        setup_experiment(config)
    data_handler = DataHandler(config)
    train_dataloader, val_dataloader = data_handler.train_loader, data_handler.val_loader
    train_dataloader = InfiniteDataloader(train_dataloader)

    trainer = Trainer(config)

    try:
        trainer.load(config.experiment.checkpoint_path)
        log.info(f'Loaded checkpoint: {config.experiment.checkpoint_path} successfully')
    except Exception as e:
        log.info(f"{e}\nCan't load checkpoint: {config.experiment.checkpoint_path}. Started from zeroth epoch")

    def prepare_batch(batch):
        data, context, weight = batch
        data, context, weight = data.to(config.utils.device),\
                                context.to(config.utils.device),\
                                weight.to(config.utils.device)
        return data, context, weight

    sampled_data, sampled_context, sampled_weight = [], [], []
    for batch in val_dataloader:
        data, context, weight = batch
        sampled_data.append(data), sampled_context.append(context), sampled_weight.append(weight)
    sampled_data = torch.cat(sampled_data, dim=0).numpy()
    sampled_context = torch.cat(sampled_context, dim=0)
    sampled_weight = torch.cat(sampled_weight, dim=0).view(-1, 1).numpy()

    wandb.log({"hist/real": wandb.Histogram(sampled_data)})

    dll_columns = ['RichDLLe', 'RichDLLk', 'RichDLLmu', 'RichDLLp', 'RichDLLbt']

    for epoch in range(0, config.experiment.epochs):
        log.info(f"Epoch {epoch} started")
        trainer.model.train()
        if config.utils.use_ddp:
            train_dataloader.loader.sampler.set_epoch(epoch)
        for iteration in tqdm(range(config.utils.epoch_iters), desc='train loop', leave=False, position=0):
            data, context, weight = prepare_batch(train_dataloader.get_next())
            if iteration == 0:
                print(f"First batch of epoch {epoch} with shapes:"
                      f" data {data.shape}, context {context.shape}, weight {weight.shape}"
                      f" Working on {config.utils.device}")
                print(f"First batch of epoch {epoch} with means:"
                      f" image {data.mean()}, context {context.mean()}, weight {weight.mean()}"
                      f" Working on {config.utils.device}")
                print(f"First batch of epoch {epoch} with stds:"
                      f" image {data.std()}, context {context.std()}, weight {weight.std()}"
                      f" Working on {config.utils.device}")
            trainer.train('C', data, context, weight)
            trainer.train('G', data, context, weight)
            # logging metrics every N iterations to save some time on uploads
            if (iteration + 1) % config.utils.log_iter_interval == 0 and main_node:
                trainer.model.eval()
                wandb.log(trainer.evaluate('C', data, context, weight, tag='training'))
                wandb.log(trainer.evaluate('G', data, context, weight, tag='training'))
                trainer.model.train()

        if (epoch + 1) % config.utils.save_interval == 0 and main_node:
            save_path = os.path.join(os.getcwd(), 'checkpoint', 'weights.{:d}.pth'.format(epoch))
            trainer.save(save_path)
            wandb.save(save_path)
            log.info(f"Checkpointed to {os.path.join(os.getcwd(), 'checkpoint', 'weights.{:d}.pth'.format(epoch))}")
            # checkpoint_fid = calculate_fid(config, trainer.model, val_dataloader, dims=2048)
            # wandb.log({'checkpoint/FID': checkpoint_fid})
            # todo add metrics

        if (epoch + 1) % config.utils.sample_interval == 0 and main_node:
            trainer.model.eval()
            with torch.no_grad():
                generated_samples = []
                for batch in val_dataloader:
                    data, context, weight = prepare_batch(batch)
                    generated_samples.append(trainer.model.generate(data, context).to('cpu'))
                generated_samples = torch.cat(generated_samples, dim=0)
                wandb.log({"hist/generated": wandb.Histogram(generated_samples)})

                fig, axes = plt.subplots(3, 2, figsize=(15, 15))
                for particle_type, ax in zip((0, 1, 2, 3, 4), axes.flatten()):
                    sns.distplot(sampled_data[:, particle_type].reshape(-1),
                                 hist_kws={'weights': sampled_weight.reshape(-1), 'alpha': 0.5},
                                 kde=False, bins=100, ax=ax, label="real normalized data", norm_hist=True)
                    sns.distplot(generated_samples[:, particle_type].reshape(-1),
                                 hist_kws={'weights': sampled_weight.reshape(-1), 'alpha': 0.5},
                                 kde=False, bins=100, ax=ax, label="generated", norm_hist=True)
                    ax.legend()
                    ax.set_title(dll_columns[particle_type])
                wandb.log({"hist/weighted_comparison": plt})
                plt.clf()

        if (epoch + 1) % config.utils.eval_interval == 0 and main_node:
            lossD, lossG = [], []
            trainer.model.eval()
            for batch in tqdm(val_dataloader, desc='val_loop', leave=False, position=0):
                data, context, weight = prepare_batch(batch)
                with torch.no_grad():
                    lossD.append(trainer.evaluate('C', data, context, weight, tag='evaluation'))
                    lossG.append(trainer.evaluate('G', data, context, weight, tag='evaluation'))
            wandb.log({key: np.mean([i[key] for i in lossD]) for key in lossD[0].keys()})
            wandb.log({key: np.mean([i[key] for i in lossG]) for key in lossG[0].keys()})
            log.info(f"Validation after epoch {epoch} finished")
    log.info("Training finished")
