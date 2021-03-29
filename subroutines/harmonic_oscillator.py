import streamlit as st

from elephant.gpfa import GPFA

from utilities.spike_train_generators import *
from utilities.spike_train_plots import *


def show_low_2d_plot(simulator):
    f1 = simulator.plot_harmonics()
    st.pyplot(f1)


def show_high_dim_plot(simulator):
    col1, col2 = st.beta_columns([2, 2])
    f2 = simulator.plot_high_dimensional()
    col1.pyplot(f2)
    f3 = simulator.plot_simulated_rasters()
    col2.pyplot(f3)


def show_gpfa_dynamics(gpfa_dynamics):
    f4 = gpfa_dynamics.plot_state_space_2d()
    st.pyplot(f4)


class harmonic_simulation:

    def __init__(self):
        st.text('')
        st.markdown('Simulate a 2D trajectory based on harmonic oscillator')
        st.text('')
        self.time_step = st.number_input('time step in milliseconds', min_value=1, max_value=100, value=1) * pq.ms
        self.trial_duration = st.number_input('total trial duration in seconds', min_value=1, max_value=10,
                                              value=2) * pq.s
        self.num_steps = int((self.trial_duration.rescale('ms') / self.time_step).magnitude)
        self.oscillator_trajectory_2dim = []
        self.times_oscillator = []

    def low_dim_trajectory(self):
        # generate a low-dimensional trajectory
        self.times_oscillator, self.oscillator_trajectory_2dim = \
            integrated_oscillator(self.time_step.magnitude, num_steps=self.num_steps, x0=0, y0=1)
        self.times_oscillator = (self.times_oscillator * self.time_step.units).rescale('s')
        simulator = simulate_plots_2d(self.oscillator_trajectory_2dim, None, None,
                                      self.times_oscillator, None, None)
        show_low_2d_plot(simulator)

    def main(self):
        self.low_dim_trajectory()
        return self.time_step, self.times_oscillator, self.oscillator_trajectory_2dim


class harmonic_spike_generation:

    def __init__(self, time_step, times_oscillator, oscillator_trajectory_2dim):
        st.text('')
        st.markdown('Simulate neurons spiking (inhomogeneous poisson) randomly based on 2D trajectory')
        st.text('')
        self.time_step = time_step
        self.times_oscillator = times_oscillator
        self.oscillator_trajectory_2dim = oscillator_trajectory_2dim
        self.max_spike_rate = st.number_input('max spike rate', min_value=5, max_value=200, value=40) * pq.Hz
        self.num_trials = st.slider('number of trials', min_value=1, max_value=50, value=20)
        self.num_spike_trains = st.slider('number of spike trains', min_value=1, max_value=50, value=20)
        self.trial_to_plot = st.number_input('Which simulated spike-train trial to plot?',
                                             min_value=0, max_value=self.num_trials - 1, value=0)
        self.spiketrains_oscillator = []

    def high_dim_trajectory(self):
        # random projection to high-dimensional space
        np.random.seed(0)
        oscillator_trajectory_ndim = random_projection(self.oscillator_trajectory_2dim,
                                                       embedding_dimension=self.num_spike_trains)
        # convert to instantaneous rate for inhomogeneous Poisson process
        normed_traj = oscillator_trajectory_ndim / oscillator_trajectory_ndim.max()
        instantaneous_rates_oscillator = np.power(self.max_spike_rate.magnitude, normed_traj)
        # generate spike trains, an object that has trial by spike train (time in seconds that spikes)
        self.spiketrains_oscillator = generate_spiketrains(instantaneous_rates_oscillator, self.num_trials,
                                                           self.time_step)
        simulator = simulate_plots_2d(self.oscillator_trajectory_2dim, self.num_spike_trains,
                                      oscillator_trajectory_ndim, self.times_oscillator, self.spiketrains_oscillator,
                                      self.trial_to_plot)
        show_high_dim_plot(simulator)

    def main(self):
        self.high_dim_trajectory()
        return self.num_trials, self.num_spike_trains, self.spiketrains_oscillator

        # TODO input Millie spiketrain here. It should be as simple as inputting the kilosort2 output (spikes in time)
        # have to bin it according to behavioral type though


class spiketrain_gpfa:

    def __init__(self, num_trials, num_spike_trains, spiketrains_oscillator):
        st.text('')
        st.markdown('Reconstruct latent dynamics using above simulated spike train')
        st.text('')
        # specify fitting parameters
        self.num_trials = num_trials
        self.num_spike_trains = num_spike_trains
        self.spiketrains_oscillator = spiketrains_oscillator
        self.bin_size = st.slider('bin size in milliseconds', min_value=5, max_value=50, value=20) * pq.ms
        self.latent_dim = st.slider('number of latent dimensions', min_value=2,
                                    max_value=self.num_spike_trains, value=2)

    def gpfa(self):
        gpfa_2dim = GPFA(bin_size=self.bin_size, x_dim=self.latent_dim)
        fitting_all = st.radio('Portion to fit GPFA', options=('First 10%', 'First half', 'All'), index=2)
        if st.button('Apply GPFA to simulated spike train', key='gpfa'):
            if fitting_all == 'First 10%':
                gpfa_2dim.fit(self.spiketrains_oscillator[:self.num_trials // 10])  # first half for training
                trajectories = gpfa_2dim.transform(self.spiketrains_oscillator)
            elif fitting_all == 'First half':
                gpfa_2dim.fit(self.spiketrains_oscillator[:self.num_trials // 2])  # first half for training
                trajectories = gpfa_2dim.transform(self.spiketrains_oscillator)
            elif fitting_all == 'All':
                trajectories = gpfa_2dim.fit_transform(self.spiketrains_oscillator)  # all
            # print(gpfa_2dim.params_estimated.keys()) # if I want to visualize params
            trial_to_plot = st.number_input('Which simulated spike-train trial to plot?',
                                            min_value=0, max_value=self.num_trials - 1, value=0, key='new_trial_num')
            gpfa_dynamics = gpfa_dynamics_plots(self.bin_size, None, trajectories, trial_to_plot,
                                                None, None)
            show_gpfa_dynamics(gpfa_dynamics)

    def main(self):
        self.gpfa()
