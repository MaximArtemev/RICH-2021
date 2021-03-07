import numpy as np

from sklearn.base import BaseEstimator
from sklearn.preprocessing import QuantileTransformer, StandardScaler, MinMaxScaler
from .utils import NoneProcessor


def get_scaler(n_quantiles):
    if n_quantiles > 0:
        return QuantileTransformer(n_quantiles=n_quantiles, output_distribution='normal', subsample=int(1e10))
    elif n_quantiles == 0:
        return StandardScaler()
    else:
        return NoneProcessor


class DataScaler(BaseEstimator):
    def __init__(self, config):
        self.config = config
        self.scalers = {
            'data': get_scaler(config.data.scaler.n_quantiles),
            'context': get_scaler(config.data.scaler.n_quantiles),
            'weight': get_scaler(config.experiment.weights.n_quantiles),
        }

        if self.config.experiment.weights.positive:
            self.scalers['weight'] = MinMaxScaler()
        if not self.config.experiment.weights.enable:
            self.scalers['weight'] = NoneProcessor()

    @staticmethod
    def _split_table(train_table, config):
        data = train_table[:, :config.experiment.data.data_dim]
        context = train_table[:, config.experiment.data.data_dim:
                                 config.experiment.data.data_dim + config.experiment.data.context_dim]
        weights = train_table[:, config.experiment.data.data_dim + config.experiment.data.context_dim:]\
            .reshape(-1, 1)
        if not config.experiment.weights.enable:
            weights = np.zeros_like(weights)
        return data, context, weights

    def fit(self, train_table):
        data, context, weights = self._split_table(train_table, self.config)
        self.scalers['data'].fit(data)
        self.scalers['context'].fit(context)
        self.scalers['weight'].fit(weights)
        return self

    def transform(self, train_table):
        data, context, weights = self._split_table(train_table, self.config)
        train_table = np.concatenate([
            self.scalers['data'].transform(data),
            self.scalers['context'].transform(context),
            self.scalers['weight'].transform(weights)
        ], axis=1)
        return train_table

    def fit_transform(self, train_table):
        return self.fit(train_table).transform(train_table)
