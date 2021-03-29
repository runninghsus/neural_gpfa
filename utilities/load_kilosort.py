import streamlit as st
import numpy as np
import pandas as pd
import os


def query_ks_workspace():
    working_dir = st.sidebar.text_input('Enter the Kilosort2 output directory:')
    try:
        os.listdir(working_dir)
        st.markdown(
            'You have selected **{}** for Kilosort2 output directory.'.format(working_dir))
    except FileNotFoundError:
        st.error('No such directory')
    return working_dir


class load_kilsort:

    def __init__(self, path):
        self.path = path

    @st.cache
    def load_spike_times(self):
        return np.load(str.join('', (self.path, '/spike_times.npy')))

    @st.cache
    def load_spike_clusters(self):
        return np.load(str.join('', (self.path, '/spike_clusters.npy')))

    @st.cache
    def load_cluster_labels(self):
        return pd.read_csv(str.join('', (self.path, '/cluster_group.tsv')), sep="\t")

    @st.cache
    def load_cluster_info(self):
        return pd.read_csv(str.join('', (self.path, '/cluster_info.tsv')), sep="\t")

    @st.cache
    def load_channel_map(self):
        return np.load(str.join('', (self.path, '/channel_map.npy')))

    @st.cache
    def load_templates(self):
        return np.load(str.join('', (self.path, '/templates.npy')))

