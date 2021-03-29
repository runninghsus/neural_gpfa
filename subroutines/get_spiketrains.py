import joblib

from utilities.load_kilosort import *
import matplotlib.pyplot as plt


class kilosort2_spike_trains:

    def __init__(self):
        self.path = query_ks_workspace()
        spike_loader = load_kilsort(self.path)
        self.spike_times = spike_loader.load_spike_times()
        self.clusters = spike_loader.load_spike_clusters()
        self.cluster_labels = spike_loader.load_cluster_labels()
        self.channel_map = spike_loader.load_channel_map()
        self.cluster_info = spike_loader.load_cluster_info()
        self.templates = spike_loader.load_templates()
        self.good_clusters = []
        try:
            with open(os.path.join(self.path, '_removed_neuron.sav'), 'rb') as fr:
                removed_neurons = joblib.load(fr)
                self.removed_neurons = removed_neurons[0]
        except FileNotFoundError:
            self.removed_neurons = []

    def find_neurons(self):
        mua = st.radio('Include MUA or not?', options=('Yes', 'No'), key='mua', index=0)
        if mua == 'Yes':
            for i, neuron in enumerate(np.unique(self.cluster_labels['cluster_id'])):
                if self.cluster_labels['group'][i] == 'good' or self.cluster_labels['group'][i] == 'mua' \
                        and neuron not in self.removed_neurons:
                    self.good_clusters.append(neuron)
        elif mua == 'No':
            for i, neuron in enumerate(np.unique(self.cluster_labels['cluster_id'])):
                if self.cluster_labels['group'][i] == 'good':
                    self.good_clusters.append(neuron)

    def show_templates(self):
        if st.checkbox('Show waveform templates?', False, key='templates'):
            fig, axs = plt.subplots(len(self.good_clusters) // 5 + 1, 5, figsize=(10, 8))
            axs = axs.ravel()
            for i, neuron in enumerate(self.good_clusters):
                axs[i].plot(self.templates[neuron][:, np.where(
                    self.channel_map[:] == self.cluster_info['channel'][neuron])[0][0]], c='black', label=neuron)
                axs[i].text(1, 0, '{}'.format(self.cluster_info['firing_rate'][neuron]), horizontalalignment='center',
                            verticalalignment='center', transform=axs[i].transAxes)
                axs[i].legend()
            st.pyplot(fig)
            if st.checkbox('Remove certain templates for analysis?', False):
                rem = st.multiselect('Which ones?', [i for i in self.good_clusters])
                self.removed_neurons.extend(rem)
                st.markdown('Click R to refresh and update.')
            if self.removed_neurons:
                with open(os.path.join(self.path, '_removed_neuron.sav'), 'wb') as f:
                    joblib.dump([self.removed_neurons], f)
            if st.button('Reset', key='reset'):
                os.remove(os.path.join(self.path, '_removed_neuron.sav'))

    def main(self):
        self.find_neurons()
        self.show_templates()
        return self.spike_times, self.clusters, self.good_clusters
