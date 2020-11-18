import streamlit as st

from elephant.gpfa import GPFA
from utilities.load_css import local_css
from utilities.spike_train_generators import *
from utilities.spike_train_plots import *

st.set_page_config(page_title='GPFA', page_icon="🕹", layout='wide', initial_sidebar_state='auto')
local_css("utilities/style.css")
title = "<div> <span class='bold'><span class='h1'>GPFA interactive</span></span> " \
        "   <span class='h2'>🕹</span> </div>"
st.markdown(title, unsafe_allow_html=True)

if st.sidebar.checkbox('Harmonic oscillator example (2D)', False):
    st.text('')
    st.markdown('Simulate a 2D trajectory based on harmonic oscillator')
    st.text('')
    time_step = st.number_input('time step in milliseconds', min_value=1, max_value=100, value=1) * pq.ms
    trial_duration = st.number_input('total trial duration in seconds', min_value=1, max_value=10, value=2) * pq.s
    num_steps = int((trial_duration.rescale('ms')/time_step).magnitude)
    # generate a low-dimensional trajectory
    times_oscillator, oscillator_trajectory_2dim = \
        integrated_oscillator(time_step.magnitude, num_steps=num_steps, x0=0, y0=1)
    times_oscillator = (times_oscillator*time_step.units).rescale('s')
    simulator = simulate_plots_2d(oscillator_trajectory_2dim, None, None,
                                  times_oscillator, None, None)
    f1 = simulator.plot_harmonics()
    st.pyplot(f1)
    st.text('')
    st.markdown('Simulate neurons spiking (inhomogeneous poisson) randomly based on 2D trajectory')
    st.text('')
    max_spike_rate = st.number_input('max spike rate', min_value=5, max_value=200, value=40) * pq.Hz
    num_trials = st.slider('number of trials', min_value=1, max_value=50, value=20)
    num_spike_trains = st.slider('number of spike trains', min_value=1, max_value=50, value=20)
    trial_to_plot = st.number_input('Which simulated spike-train trial to plot?',
                                    min_value=0, max_value=num_trials-1, value=0)
    # random projection to high-dimensional space
    np.random.seed(0)
    oscillator_trajectory_ndim = random_projection(oscillator_trajectory_2dim, embedding_dimension=num_spike_trains)
    # convert to instantaneous rate for inhomogeneous Poisson process
    normed_traj = oscillator_trajectory_ndim / oscillator_trajectory_ndim.max()
    instantaneous_rates_oscillator = np.power(max_spike_rate.magnitude, normed_traj)
    # generate spike trains, an object that has trial by spike train (time in seconds that spikes)
    spiketrains_oscillator = generate_spiketrains(instantaneous_rates_oscillator, num_trials, time_step)
    simulator = simulate_plots_2d(oscillator_trajectory_2dim, num_spike_trains, oscillator_trajectory_ndim,
                                  times_oscillator, spiketrains_oscillator, trial_to_plot)
    col2, col3 = st.beta_columns([2, 2])
    f2 = simulator.plot_high_dimensional()
    col2.pyplot(f2)
    f3 = simulator.plot_simulated_rasters()
    col3.pyplot(f3)

    # TODO input Millie spiketrain here. It should be as simple as inputting the kilosort2 output (spikes in time)
    # have to bin it according to behavioral type though
    st.text('')
    st.markdown('Reconstruct latent dynamics using above simulated spike train')
    st.text('')
    # specify fitting parameters
    bin_size = st.slider('bin size in milliseconds', min_value=5, max_value=50, value=20) * pq.ms
    latent_dim = st.slider('number of latent dimensions', min_value=2, max_value=num_spike_trains, value=2)
    gpfa_2dim = GPFA(bin_size=bin_size, x_dim=latent_dim)
    fitting_all = st.radio('Portion to fit GPFA', options=('First 10%', 'First half', 'All'), index=2)
    if st.button('Apply GPFA to simulated spike train', key='gpfa'):
        if fitting_all == 'First 10%':
            gpfa_2dim.fit(spiketrains_oscillator[:num_trials // 10])  # first half for training
            trajectories = gpfa_2dim.transform(spiketrains_oscillator)
        elif fitting_all == 'First half':
            gpfa_2dim.fit(spiketrains_oscillator[:num_trials//2])  # first half for training
            trajectories = gpfa_2dim.transform(spiketrains_oscillator)
        elif fitting_all == 'All':
            trajectories = gpfa_2dim.fit_transform(spiketrains_oscillator)  # al
        # print(gpfa_2dim.params_estimated.keys()) # if I want to visualize params
        trial_to_plot = st.number_input('Which simulated spike-train trial to plot?',
                                        min_value=0, max_value=num_trials - 1, value=0, key='new_trial_num')
        gpfa_dynamics = gpfa_dynamics_plots(bin_size, oscillator_trajectory_2dim, trajectories, trial_to_plot,
                                            None, None)
        f4 = gpfa_dynamics.plot_state_space_2d()
        st.pyplot(f4)

if st.sidebar.checkbox('Lorentz system example (3D)', False):
    st.text('')
    st.markdown('Simulate a 3D trajectory based on Lorentz system')
    st.text('')
    time_step = st.number_input('time step in milliseconds', min_value=1, max_value=100, value=1) * pq.ms
    transient_duration = st.number_input('transient duration in seconds', min_value=1, max_value=10, value=10) * pq.s
    trial_duration = st.number_input('total trial duration in seconds', min_value=1, max_value=30, value=30) * pq.s
    num_steps_transient = int((transient_duration.rescale('ms') / time_step).magnitude)
    num_steps = int((trial_duration.rescale('ms')/time_step).magnitude)
    # calculate the oscillator
    times, lorentz_trajectory_3dim = integrated_lorenz(time_step, num_steps=num_steps_transient + num_steps,
                                                       x0=0, y0=1, z0=1.25)
    times = (times - transient_duration).rescale('s').magnitude
    times_trial = times[num_steps_transient:]
    simulator = simulate_plots_3d(times, lorentz_trajectory_3dim, transient_duration, num_steps_transient)
    f = simulator.plot_lorentz()
    st.pyplot(f)
    st.text('')
    st.markdown('Simulate neurons spiking (inhomogeneous poisson) randomly based on 3D trajectory')
    st.text('')
    max_spike_rate = st.number_input('max spike rate', min_value=5, max_value=200, value=40) * pq.Hz
    num_trials = st.slider('number of trials', min_value=1, max_value=50, value=20)
    num_spike_trains = st.slider('number of spike trains', min_value=1, max_value=50, value=20)
    trial_to_plot = st.number_input('Which simulated spike-train trial to plot?',
                                    min_value=0, max_value=num_trials-1, value=0)
    # random projection
    np.random.seed(0)
    lorentz_trajectory_ndim = random_projection(lorentz_trajectory_3dim[:, num_steps_transient:],
                                                embedding_dimension=num_spike_trains)
    # calculate instantaneous rate
    normed_traj = lorentz_trajectory_ndim / lorentz_trajectory_ndim.max()
    instantaneous_rates_lorentz = np.power(max_spike_rate.magnitude, normed_traj)
    # generate spiketrains
    spiketrains_lorentz = generate_spiketrains(instantaneous_rates_lorentz, num_trials, time_step)
    simulator = simulate_plots_2d(None, num_spike_trains, lorentz_trajectory_ndim,
                                  times_trial, spiketrains_lorentz, trial_to_plot)
    col2, col3 = st.beta_columns([2, 2])
    f2 = simulator.plot_high_dimensional()
    col2.pyplot(f2)
    f3 = simulator.plot_simulated_rasters()
    col3.pyplot(f3)
    st.text('')
    st.markdown('Reconstruct latent dynamics using above simulated spike trains')
    st.text('')
    # specify fitting parameters
    bin_size = st.slider('bin size in milliseconds', min_value=5, max_value=50, value=20) * pq.ms
    latent_dim = st.slider('number of latent dimensions', min_value=3, max_value=num_spike_trains, value=3)
    gpfa_3dim = GPFA(bin_size=bin_size, x_dim=latent_dim)
    fitting_all = st.radio('Portion to fit GPFA', options=('First 10%', 'First half', 'All'), index=2)
    if st.button('Apply GPFA to simulated spike train', key='gpfa'):
        if fitting_all == 'First 10%':
            gpfa_3dim.fit(spiketrains_lorentz[:num_trials // 10])  # first half for training
            trajectories = gpfa_3dim.transform(spiketrains_lorentz)
        elif fitting_all == 'First half':
            gpfa_3dim.fit(spiketrains_lorentz[:num_trials//2])  # first half for training
            trajectories = gpfa_3dim.transform(spiketrains_lorentz)
        elif fitting_all == 'All':
            trajectories = gpfa_3dim.fit_transform(spiketrains_lorentz)  # al
        # trial_to_plot = st.number_input('Which simulated spike-train trial to plot?',
        #                                 min_value=0, max_value=num_trials - 1, value=0, key='new_trial_num')
        gpfa_dynamics = gpfa_dynamics_plots(bin_size, lorentz_trajectory_3dim, trajectories, None,
                                            num_steps_transient, times_trial)
        f4 = gpfa_dynamics.plot_state_space_3d()
        st.pyplot(f4)