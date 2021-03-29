from subroutines.lorentz_system import *
from utilities.processing import *

class bsoid_gpfa:

    def __init__(self, spike_times, clusters, predictions):

        self.bin_size = st.slider('bin size in milliseconds', min_value=5, max_value=50, value=20) * pq.ms
        self.latent_dim = st.slider('number of latent dimensions', min_value=3, max_value=10, value=3)
        self.gpfa_3dim = GPFA(bin_size=self.bin_size, x_dim=self.latent_dim)
        self.spike_times = spike_times
        self.clusters = clusters
        self.predictions = predictions
        self.spike_time_idx = []
        self.num_trials = []
        self.trajectories = []
        self.spike_trains_trials = []
        self.term_frame = []
        self.bout_frames = []

    def correlates(self):
        for group_num in np.unique(self.predictions):
            self.bout_frames.append(np.array(np.where(self.predictions == group_num)))
            term_f = np.diff(self.bout_frames) != 1
            self.term_frame.append(np.array(term_f * 1))
            lengths, pos, grp = rle(self.term_frame.T)




        for i in range(len(np.unique(self.predictions))):
            behavior_idx = np.where(self.predictions == i)

            #TODO do the diff method like we did in matlab/kinematics
            for an, se in enumerate(np.concatenate(exp)):
                bout_frames.append(np.array(np.where(fs_labels[se] == group_num)))
                term_f = np.diff(bout_frames[an]) != 1
                term_frame.append(np.array(term_f * 1))
                lengths, pos, grp = rle(term_frame[an].T)
                endpos = np.where(np.diff(pos) < 1)[0][0] + 1


    def neo_spike_trains(self):
        for _ in range(self.num_trials):
            spike_trains_trial = []
            for i in range(1, len(self.spike_time_idx)):
                spike_times_s = self.spike_times[self.spike_time_idx[i]] / 30000
            try:
                spike_trains_trial.append(neo.SpikeTrain(np.concatenate(spike_times_s), units='sec', t_stop=30))
            except:
                pass
        self.spike_trains.append(spike_trains_trial)

    def fit(self):
        self.trajectories = self.gpfa_3dim.fit_transform(self.spike_trains)

    def main(self):
        self.correlates()
        self.neo_spike_trains()
        if st.button('Extract latent dynamics'):
            self.fit()
            gpfa_dynamics = gpfa_dynamics_plots(self.bin_size, None, self.trajectories, None, None, None)
            f4 = gpfa_dynamics.plot_state_space_3d()
            st.pyplot(f4)
