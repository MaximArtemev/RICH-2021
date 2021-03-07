import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

dll_columns = ['RichDLLe', 'RichDLLk', 'RichDLLmu', 'RichDLLp', 'RichDLLbt']
raw_feature_columns = ['Brunel_P', 'Brunel_ETA', 'nTracks_Brunel']
weight_col = 'probe_sWeight'
list_particles = ['kaon', 'pion', 'proton', 'muon', 'electron']


def load_and_cut(file_name):
    data = pd.read_csv(file_name, delimiter='\t')
    return data[dll_columns + raw_feature_columns + [weight_col]]


def load_and_merge_and_cut(filename_list):
    return pd.concat([load_and_cut(fname) for fname in filename_list], axis=0, ignore_index=True)


def get_particle_table(data_path, particle):
    particle_files = [os.path.join(data_path, name) for name in os.listdir(data_path) if particle in name]
    particle_csv = []
    for path in particle_files:
        table = pd.read_csv(path, delimiter='\t')
        table = table[dll_columns + raw_feature_columns + [weight_col]]
        particle_csv.append(table)
    particle_csv = pd.concat(particle_csv, axis=0, ignore_index=True)
    return particle_csv


class NoneProcessor(StandardScaler):
    def fit(self, X, y=None):
        pass

    def transform(self, X, copy=None):
        return X

    def inverse_transform(self, X, copy=None):
        return X
