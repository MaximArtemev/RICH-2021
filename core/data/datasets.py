import torch
import logging

log = logging.getLogger(__name__)


class ParticleDataset(torch.utils.data.Dataset):
    def __init__(self, config, table):
        self.data = table[:, :config.experiment.data.data_dim]
        self.context = table[:, config.experiment.data.data_dim:
                                     config.experiment.data.data_dim + config.experiment.data.context_dim]
        self.weight = table[:, -1]
        assert config.experiment.data.data_dim + config.experiment.data.context_dim + 1 == table.shape[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.context[idx], self.weight[idx]
