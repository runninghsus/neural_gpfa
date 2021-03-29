import streamlit as st

from elephant.gpfa import GPFA

from subroutines.harmonic_oscillator import show_high_dim_plot
from utilities.spike_train_generators import *
from utilities.spike_train_plots import *


def show_low_3d_plot(simulator):
    f1 = simulator.plot_lorentz()
    st.pyplot(f1)


def show_3d_gpfa_dynamics(gpfa_dynamics):
    f4 = gpfa_dynamics.plot_state_space_3d()
    st.pyplot(f4)


class lorentz_simulation:

    def __init__(self):
        st.text('')
        st.markdown('Simulate a 3D trajectory based on Lorentz system')
        st.text('')
        self.time_step = st.number_input('time step in milliseconds', min_value=1, max_value=100, value=1) * pq.ms
        self.transient_duration = st.number_input('transient duration in seconds', min_value=1, max_value=30,
                                                  value=10) * pq.s
        self.trial_duration = st.number_input('total trial duration in seconds', min_value=1, max_value=90,
                                              value=30) * pq.s
        self.num_steps_transient = int((self.transient_duration.rescale('ms') / self.time_step).magnitude)
        self.num_steps = int((self.trial_duration.rescale('ms') / self.time_step).magnitude)
        self.times = []
        self.lorentz_trajectory_3dim = []

    def low_dim_trajectory(self):
        # calculate the oscillator
        self.times, self.lorentz_trajectory_3dim = \
            integrated_lorenz(self.time_step, num_steps=self.num_steps_transient + self.num_steps, x0=0, y0=1, z0=1.25)
        self.times = (self.times - self.transient_duration).rescale('s').magnitude
        self.times_trial = self.times[self.num_steps_transient:]
        simulator = simulate_plots_3d(self.times, self.lorentz_trajectory_3dim, self.transient_duration,
                                      self.num_steps_transient)
        show_low_3d_plot(simulator)

    def main(self):
        self.low_dim_trajectory()
        return self.time_step, self.num_steps_transient, self.times_trial, self.times, self.lorentz_trajectory_3dim


class lorentz_spike_generation:

    def __init__(self, time_step, num_steps_transient, times_trial, times, lorentz_trajectory_3dim):
        st.text('')
        st.markdown('Simulate neurons spiking (inhomogeneous poisson) randomly based on 3D trajectory')
        st.text('')
        self.time_step = time_step
        self.num_steps_transient = num_steps_transient
        self.times_trial = times_trial
        self.times = times
        self.lorentz_trajectory_3dim = lorentz_trajectory_3dim
        self.max_spike_rate = st.number_input('max spike rate', min_value=5, max_value=200, value=40) * pq.Hz
        self.num_trials = st.slider('number of trials', min_value=1, max_value=50, value=20)
        self.num_spike_trains = st.slider('number of spike trains', min_value=1, max_value=50, value=20)
        self.trial_to_plot = st.number_input('Which simulated spike-train trial to plot?',
                                             min_value=0, max_value=self.num_trials - 1, value=0)
        self.spiketrains_lorentz = []
        self.lorentz_trajectory_ndim = []

    def high_dim_trajectory(self):
        # random projection to high-dimensional space
        np.random.seed(0)
        self.lorentz_trajectory_ndim = random_projection(self.lorentz_trajectory_3dim[:, self.num_steps_transient:],
                                                         embedding_dimension=self.num_spike_trains)
        # calculate instantaneous rate
        normed_traj = self.lorentz_trajectory_ndim / self.lorentz_trajectory_ndim.max()
        instantaneous_rates_lorentz = np.power(self.max_spike_rate.magnitude, normed_traj)
        # generate spiketrains
        self.spiketrains_lorentz = generate_spiketrains(instantaneous_rates_lorentz, self.num_trials, self.time_step)
        simulator = simulate_plots_2d(None, self.num_spike_trains, self.lorentz_trajectory_ndim,
                                      self.times_trial, self.spiketrains_lorentz, self.trial_to_plot)
        show_high_dim_plot(simulator)

    def main(self):
        self.high_dim_trajectory()
        return self.num_trials, self.times_trial, self.num_steps_transient, self.num_spike_trains, self.spiketrains_lorentz


class spiketrain_3d_gpfa:

    def __init__(self, num_trials, times_trial, num_steps_transient, num_spike_trains, spiketrains_lorentz):
        st.text('')
        st.markdown('Reconstruct latent dynamics using above simulated spike train')
        st.text('')
        # specify fitting parameters
        self.num_trials = num_trials
        self.times_trial = times_trial
        self.num_steps_transient = num_steps_transient
        self.num_spike_trains = num_spike_trains
        self.spiketrains_lorentz = spiketrains_lorentz
        self.bin_size = st.slider('bin size in milliseconds', min_value=5, max_value=50, value=20) * pq.ms
        self.latent_dim = st.slider('number of latent dimensions', min_value=3, max_value=num_spike_trains, value=3)

    def gpfa(self):
        gpfa_3dim = GPFA(bin_size=self.bin_size, x_dim=self.latent_dim)
        fitting_all = st.radio('Portion to fit GPFA', options=('First 10%', 'First half', 'All'), index=2)
        if st.button('Apply GPFA to simulated spike train', key='gpfa'):
            if fitting_all == 'First 10%':
                gpfa_3dim.fit(self.spiketrains_lorentz[:self.num_trials // 10])  # first half for training
                trajectories = gpfa_3dim.transform(self.spiketrains_lorentz)
            elif fitting_all == 'First half':
                gpfa_3dim.fit(self.spiketrains_lorentz[:self.num_trials // 2])  # first half for training
                trajectories = gpfa_3dim.transform(self.spiketrains_lorentz)
            elif fitting_all == 'All':
                trajectories = gpfa_3dim.fit_transform(self.spiketrains_lorentz)  # all
            # print(gpfa_3dim.params_estimated.keys()) # if I want to visualize params
            gpfa_dynamics = gpfa_dynamics_plots(self.bin_size, None, trajectories, None,
                                                self.num_steps_transient, self.times_trial)
            show_3d_gpfa_dynamics(gpfa_dynamics)

    def main(self):
        self.gpfa()