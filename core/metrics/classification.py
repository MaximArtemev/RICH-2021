import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from omegaconf.dictconfig import DictConfig
from sklearn.model_selection import train_test_split
from typing import Tuple


def weighted_roc_auc(classes: np.array,
                     predictions: np.array,
                     weights: np.array = None) -> float:

    if weights is None:
        weights = np.ones_like(predictions)

    assert len(classes) == len(predictions) == len(weights)
    assert classes.ndim == predictions.ndim == weights.ndim == 1
    class0, class1 = sorted(np.unique(classes))

    data = np.empty(
        shape=len(classes),
        dtype=[('c', classes.dtype),
               ('p', predictions.dtype),
               ('w', weights.dtype)]
    )
    data['c'], data['p'], data['w'] = classes, predictions, weights

    data = data[np.argsort(data['c'])]
    data = data[
        np.argsort(data['p'], kind='mergesort')]  # here we're relying on stability as we need class orders preserved

    correction = 0.
    # mask1 - bool mask to highlight collision areas
    # mask2 - bool mask with collision areas' start points
    mask1 = np.empty(len(data), dtype=bool)
    mask2 = np.empty(len(data), dtype=bool)
    mask1[0] = mask2[-1] = False
    mask1[1:] = data['p'][1:] == data['p'][:-1]
    if mask1.any():
        mask2[:-1] = ~mask1[:-1] & mask1[1:]
        mask1[:-1] |= mask1[1:]
        ids, = mask2.nonzero()
        correction = sum([((dsplit['c'] == class0) * dsplit['w'] * msplit).sum() *
                          ((dsplit['c'] == class1) * dsplit['w'] * msplit).sum()
                          for dsplit, msplit in zip(np.split(data, ids), np.split(mask1, ids))]) * 0.5

    weights_0 = data['w'] * (data['c'] == class0)
    weights_1 = data['w'] * (data['c'] == class1)
    cumsum_0 = weights_0.cumsum()

    return ((cumsum_0 * weights_1).sum() - correction) / (weights_1.sum() * cumsum_0[-1])


def calculate_rocauc(
        config: DictConfig,
        features: pd.DataFrame,
        targets_real: pd.DataFrame,
        targets_fake: pd.DataFrame,
        weights: np.array) -> Tuple[float, float]:
    classifier = CatBoostClassifier(
        iterations=config.metric.classification.iterations,
        depth=config.metric.classification.depth,
        task_type="GPU" if 'cuda' in config.utils.device else None,
        devices=config.utils.device[-1] if 'cuda' in config.utils.device else None
    )
    data = pd.concat([
        pd.concat([targets_real, features], axis=1),
        pd.concat([targets_fake, features], axis=1)
    ], axis=0)
    labels = np.concatenate([
        np.ones(targets_real.shape[0]),
        np.zeros(targets_fake.shape[0])
    ], axis=0)
    weights = np.concatenate([
        weights,
        weights,
    ], axis=0)
    train_table, val_table, \
    train_labels, val_labels, \
    _, val_weights = train_test_split(data, labels.reshape(-1), weights.reshape(-1),
                                      test_size=config.metric.classification.split_size)
    classifier.fit(train_table, train_labels, verbose=config.metric.classification.verbose_each_iter)
    predictions = classifier.predict_proba(val_table)[:, 1]
    val_labels = val_labels.reshape(-1)
    predictions = predictions.reshape(-1)
    val_weights = val_weights.reshape(-1)
    weighted = weighted_roc_auc(val_labels, predictions, val_weights)
    unweighted = weighted_roc_auc(val_labels, predictions, np.ones_like(val_weights))
    return weighted, unweighted
