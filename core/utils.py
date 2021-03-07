import torch


class LossWeighter:
    def __init__(self, config):
        self.config = config
        self.weights = DotDict(config.losses)

    def combine_losses(self, pairs):
        combined_loss = 0
        for loss_name, loss in pairs.items():
            if self.weights._getattr(loss_name) != 0:
                combined_loss += self.weights._getattr(loss_name) * loss
        return combined_loss

    def visualize_losses(self, pairs, mode):
        visualized_losses = {}
        for loss_name, loss in pairs.items():
            text = f"{mode}/{'/'.join(loss_name.split('.'))}"
            visualized_losses[text] = loss.item()
        return visualized_losses


class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value

    def _getattr(self, key):
        target = self
        for dot in key.split('.'):
            target = target[dot]
        return target

    def _setattr(self, key, value):
        target = self
        for dot in key.split('.')[:-1]:
            target = target[dot]
        target[key.split('.')[-1]] = value


class InfiniteDataloader:
    def __init__(self, loader):
        self.loader = loader
        self.iter = iter(self.loader)

    def get_next(self):
        try:
            return next(self.iter)
        except StopIteration:
            self.iter = iter(self.loader)
            return self.get_next()
