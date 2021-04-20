import os
import numpy as np
from sklearn.model_selection import train_test_split
from omegaconf.dictconfig import DictConfig

import torch.utils.data as data

from .datasets import ParticleDataset
from .transformer import DataScaler
from .utils import get_particle_table
import logging

log = logging.getLogger(__name__)


class DataHandler:
    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.scaler = DataScaler(config)

        if config.data.download:
            if not os.path.exists(config.data.data_path):
                os.makedirs(config.data.data_path)
            log.info('config.data.download is True, starting dowload')
            target_path = os.path.join(config.data.data_path, 'data_calibsample')
            if os.path.exists(target_path):
                print("It seems that data is already downloaded. Are you sure?")
            os.system(f"wget https://cernbox.cern.ch/index.php/s/Fjf3UNgvlRVa4Td/download -O {target_path + '.tar.gz'}")
            log.info('files downloaded, starting unpacking')
            os.system(f"tar xvf {target_path + '.tar.gz'} -C {config.data.data_path}")
            log.info('files unpacked')

        config.data.data_path = os.path.join(config.data.data_path, 'data_calibsample')

        table = np.array(get_particle_table(config.data.data_path, config.experiment.particle))
        train_table, val_table = train_test_split(table, test_size=self.config.data.val_size, random_state=42)
        train_table = self.scaler.fit_transform(train_table)
        val_table = self.scaler.transform(val_table)

        self.train_loader = data.DataLoader(
            dataset=ParticleDataset(config, train_table),
            batch_size=config.experiment.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True
        )
        self.val_loader = data.DataLoader(
            dataset=ParticleDataset(config, val_table),
            batch_size=config.experiment.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=True
        )
