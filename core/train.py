import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf.dictconfig import DictConfig
from typing import List, Tuple
from torchtyping import TensorType, patch_typeguard
from core.utils import DataTensorType, WeightTensorType, ContextTensorType
from typeguard import typechecked

import torch
import wandb

from core.data import DataHandler
from core.utils import InfiniteDataloader
from core.trainer import Trainer
# from core.metrics import calculate_fid
# todo add fid

import logging
from omegaconf import OmegaConf

log = logging.getLogger(__name__)
patch_typeguard()


def setup_experiment(config: DictConfig) -> None:
    os.environ["WANDB_API_KEY"] = config.wandb.api_key
    wandb.login()
    wandb.init(project=config.wandb.project_name,
               config=OmegaConf.to_container(config, resolve=True), 
               mode='online' if config.wandb.enabled else 'disabled')
    wandb.config.run_dir = os.getcwd()
    save_path = os.path.join(os.getcwd(), '.hydra', 'config.yaml')
    wandb.save(save_path)


def train(config: DictConfig) -> None:
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
    inverted_sampled_data = data_handler.scaler.scalers['data'].inverse_transform(sampled_data)
    sampled_context = torch.cat(sampled_context, dim=0)
    sampled_weight = torch.cat(sampled_weight, dim=0).view(-1, 1).numpy()

    dll_columns = ['RichDLLe', 'RichDLLk', 'RichDLLmu', 'RichDLLp', 'RichDLLbt']

    for epoch in range(0, config.experiment.epochs):
        log.info(f"Epoch {epoch} started")
        trainer.model.train()
        for iteration in tqdm(range(config.utils.epoch_iters), desc='train loop', leave=False, position=0):
            data, context, weight = prepare_batch(train_dataloader.get_next())

            if iteration == 0:
                log.info(f"First batch of epoch {epoch} with:"
                         f" shapes: {data.shape}, {context.shape}, {weight.shape}"
                         f" means: {data.mean()}, {context.mean()}, {weight.mean()}"
                         f" stds: {data.std()}, {context.std()}, {weight.std()}")
            trainer.train('C', data, context, weight)
            trainer.train('G', data, context, weight)
            # logging metrics every N iterations to save some time on uploads
            if (iteration + 1) % config.utils.log_iter_interval == 0:
                trainer.model.eval()
                wandb.log(trainer.evaluate('C', data, context, weight, tag='training'))
                wandb.log(trainer.evaluate('G', data, context, weight, tag='training'))
                trainer.model.train()

        if (epoch + 1) % config.utils.save_interval == 0:
            save_path = os.path.join(os.getcwd(), 'checkpoint', 'weights.{:d}.pth'.format(epoch))
            trainer.save(save_path)
            wandb.save(save_path)
            log.info(f"Checkpointed to {os.path.join(os.getcwd(), 'checkpoint', 'weights.{:d}.pth'.format(epoch))}")
            # checkpoint_fid = calculate_fid(config, trainer.model, val_dataloader, dims=2048)
            # wandb.log({'checkpoint/FID': checkpoint_fid})
            # todo add metrics

        if (epoch + 1) % config.utils.sample_interval == 0:
            trainer.model.eval()
            with torch.no_grad():
                generated_samples = []
                for batch in val_dataloader:
                    data, context, weight = prepare_batch(batch)
                    generated_samples.append(trainer.model.generate(data, context).to('cpu'))
                generated_samples = torch.cat(generated_samples, dim=0).numpy()
                inverted_generated_samples = data_handler.scaler.scalers['data'].inverse_transform(generated_samples)

                hist_kws = {
                    'weights': data_handler.scaler.scalers['weight'].inverse_transform(sampled_weight).reshape(-1),
                    'alpha': 0.5
                }

                fig, axes = plt.subplots(3, 2, figsize=(15, 15))
                for particle_type, ax in zip((0, 1, 2, 3, 4), axes.flatten()):
                    sns.distplot(
                        inverted_sampled_data[:, particle_type].reshape(-1),
                        hist_kws=hist_kws,
                        kde=False, bins=100, ax=ax, label="real normalized data", norm_hist=True
                    )
                    sns.distplot(
                        inverted_generated_samples[:, particle_type].reshape(-1),
                        hist_kws=hist_kws,
                        kde=True, bins=100, ax=ax, label="generated", norm_hist=True
                    )
                    ax.legend()
                    ax.set_title(dll_columns[particle_type])
                wandb.log({"hist/real-weighted": wandb.Image(plt)})
                plt.clf()

                hist_kws = {
                    'weights': np.ones_like(sampled_weight).reshape(-1),
                    'alpha': 0.5
                }

                fig, axes = plt.subplots(3, 2, figsize=(15, 15))
                for particle_type, ax in zip((0, 1, 2, 3, 4), axes.flatten()):
                    sns.distplot(
                        inverted_sampled_data[:, particle_type].reshape(-1),
                        hist_kws=hist_kws,
                        kde=False, bins=100, ax=ax, label="real normalized data", norm_hist=True
                    )
                    sns.distplot(
                        inverted_generated_samples[:, particle_type].reshape(-1),
                        hist_kws=hist_kws,
                        kde=True, bins=100, ax=ax, label="generated", norm_hist=True
                    )
                    ax.legend()
                    ax.set_title(dll_columns[particle_type])
                wandb.log({"hist/real-unweighted": wandb.Image(plt)})
                plt.clf()

        if (epoch + 1) % config.utils.eval_interval == 0:
            lossD, lossG = [], []
            trainer.model.eval()
            for batch_ind, batch in enumerate(tqdm(val_dataloader, desc='val_loop', leave=False, position=0)):
                if batch_ind > config.utils.epoch_iters:
                    break
                data, context, weight = prepare_batch(batch)
                with torch.no_grad():
                    lossD.append(trainer.evaluate('C', data, context, weight, tag='evaluation'))
                    lossG.append(trainer.evaluate('G', data, context, weight, tag='evaluation'))
            wandb.log({key: np.mean([i[key] for i in lossD]) for key in lossD[0].keys()})
            wandb.log({key: np.mean([i[key] for i in lossG]) for key in lossG[0].keys()})
            log.info(f"Validation after epoch {epoch} finished")
    log.info("Training finished")
