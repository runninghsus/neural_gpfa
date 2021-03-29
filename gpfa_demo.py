from subroutines.harmonic_oscillator import *
from subroutines.lorentz_system import *
from utilities.load_css import local_css

st.set_page_config(page_title='GPFA demo', page_icon="🕹", layout='wide', initial_sidebar_state='auto')
local_css("utilities/style.css")
title = "<div> <span class='bold'><span class='h1'>GPFA demo</span></span> " \
        "   <span class='h2'>🕹</span> </div>"
st.markdown(title, unsafe_allow_html=True)

if st.sidebar.checkbox('Harmonic oscillator example (2D)', False):
    harmonics = harmonic_simulation()
    time_step, times_oscillator, oscillator_trajectory_2dim = harmonics.main()
    spikes_harmonics = harmonic_spike_generation(time_step, times_oscillator, oscillator_trajectory_2dim)
    num_trials, num_spike_trains, spiketrains_oscillator = spikes_harmonics.main()
    latent_dynamics = spiketrain_gpfa(num_trials, num_spike_trains, spiketrains_oscillator)
    latent_dynamics.main()

if st.sidebar.checkbox('Lorentz system example (3D)', False):
    lorentz = lorentz_simulation()
    time_step, num_steps_transient, times_trial, times, lorentz_trajectory_3dim = lorentz.main()
    spikes_lorentz = lorentz_spike_generation(time_step, num_steps_transient,
                                              times_trial, times, lorentz_trajectory_3dim)
    num_trials, times_trial, num_steps_transient, num_spike_trains, spiketrains_lorentz = spikes_lorentz.main()
    latent_3d_dynamics = spiketrain_3d_gpfa(num_trials, times_trial, num_steps_transient,
                                            num_spike_trains, spiketrains_lorentz)
    latent_3d_dynamics.main()
