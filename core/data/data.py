import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.model_selection import train_test_split

import torch.utils.data as data

from .datasets import ParticleDataset
from .utils import get_particle_table
import logging

log = logging.getLogger(__name__)


class DataHandler:
    def __init__(self, config):
        self.config = config

        if config.data.scaler.n_quantiles > 0:
            self.scaler = QuantileTransformer(
                n_quantiles=config.data.scaler.n_quantiles,
                output_distribution='normal',
                subsample=int(1e10)
            )
        else:
            self.scaler = StandardScaler()

        if config.data.download:
            assert os.path.exists(config.data.data_path)
            log.info('config.data.download is True, starting dowload')
            target_path = os.path.join(config.data.data_path, 'data_calibsample.tar.gz')
            if os.path.exists(target_path):
                print("It seems that data is already downloaded. Are you sure?")
            os.system(f"wget https://cernbox.cern.ch/index.php/s/Fjf3UNgvlRVa4Td/download -O {target_path}")
            log.info('files downloaded, starting unpacking')
            os.system(f"tar xvf {target_path}")
            log.info('files unpacked')

        # todo rethink
        config.data.data_path = os.path.join(config.data.data_path, 'data_calibsample')

        table = np.array(get_particle_table(config.data.data_path, config.experiment.particle))
        train_table, val_table = train_test_split(table, test_size=self.config.data.val_size, random_state=42)
        self.scaler.fit(train_table[:, :-1]) # without the weights
        # todo assert weight on last col

        train_table = np.concatenate([self.scaler.transform(train_table[:, :-1]), train_table[:, -1].reshape(-1, 1)], axis=1)
        val_table = np.concatenate([self.scaler.transform(val_table[:, :-1]), val_table[:, -1].reshape(-1, 1)], axis=1)

        train_dataset = ParticleDataset(config, train_table)
        self.train_loader = data.DataLoader(
            dataset=train_dataset,
            batch_size=config.experiment.batch_size,
            sampler=data.DistributedSampler(train_dataset) if config.utils.use_ddp else None,
            shuffle=True if not config.utils.use_ddp else None,
            pin_memory=True,
            drop_last=True
        )
        val_dataset = ParticleDataset(config, val_table)
        self.val_loader = data.DataLoader(
            dataset=val_dataset,
            batch_size=config.experiment.batch_size,
            sampler=None,
            shuffle=False,
            drop_last=True
        )