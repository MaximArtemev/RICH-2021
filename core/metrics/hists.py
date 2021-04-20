import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from core.data import dll_columns


def plot_1d_hist(real_data: np.array,
                 fake_data: np.array,
                 hist_kws: dict = None) -> plt.Figure:
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    for particle_type, ax in zip((0, 1, 2, 3, 4), axes.flatten()):
        sns.distplot(
            real_data[:, particle_type].reshape(-1),
            hist_kws=hist_kws,
            kde=False, bins=100, ax=ax, label="real data", norm_hist=True
        )
        sns.distplot(
            fake_data[:, particle_type].reshape(-1),
            hist_kws=hist_kws,
            kde=True, bins=100, ax=ax, label="generated data", norm_hist=True
        )
        ax.legend()
        ax.set_title(dll_columns[particle_type])
    return fig
