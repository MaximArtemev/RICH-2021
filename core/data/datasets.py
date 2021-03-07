from torch.utils.data import Dataset
import logging

from .transformer import DataScaler

log = logging.getLogger(__name__)


class ParticleDataset(Dataset):
    def __init__(self, config, table):
        self.data, self.context, self.weight = DataScaler._split_table(table, config)
        assert config.experiment.data.data_dim + config.experiment.data.context_dim + 1 == table.shape[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.context[idx], self.weight[idx]
