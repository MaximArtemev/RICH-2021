import numpy as np
from torch.utils.data import Dataset
import logging
from typing import Tuple
from omegaconf.dictconfig import DictConfig

from .transformer import DataScaler

log = logging.getLogger(__name__)


class ParticleDataset(Dataset):
    def __init__(self, config:DictConfig, table: np.array) -> None:
        self.data, self.context, self.weight = DataScaler._split_table(table, config)
        assert config.experiment.data.data_dim + config.experiment.data.context_dim + 1 == table.shape[1]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[np.array, np.array, np.array]:
        return self.data[idx], self.context[idx], self.weight[idx]
