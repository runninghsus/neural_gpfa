from subroutines.get_behavior import *
from subroutines.get_spiketrains import *
from subroutines.neural_statespace import *
from utilities.load_css import local_css
import random
from sklearn.model_selection import cross_val_score


st.set_page_config(page_title='G-SOiD v1', page_icon="➿", layout='wide', initial_sidebar_state='auto')
local_css("utilities/style.css")
title = "<div> <span class='bold'><span class='h1'>G-SOiD</span></span> " \
        "   <span class='h2'> --version 1.0 ➿</span> </div>"
st.markdown(title, unsafe_allow_html=True)
st.markdown('Step 1: Fill in Kilosort2 output directory.')
st.markdown('Step 2: Fill in B-SOiD output directory.')
st.markdown('Step 3: Starting with preprocess and alignment, select single procedure and progress.')
st.text('')

if st.sidebar.checkbox('Load spiketrains from Kilosort2', key='kilosort2'):
    spike_trains_loader = kilosort2_spike_trains()
    spike_times, clusters, good_clusters = spike_trains_loader.main()

if st.sidebar.checkbox('Load behaviors from B-SOiD', key='bsoid'):
    behavior_loader = bsoid_behavior()
    f_index, new_predictions = behavior_loader.main()

if st.sidebar.checkbox('Alignment and preprocess'):
    st.text('')
    st.markdown('Neural alignment')
    st.text('')
    sample_rate = st.number_input('Neural sample rate', min_value=10000, max_value=None, value=30000)
    neural_start_sample = st.number_input('Start sample for neural behavioral',
                                          min_value=0, max_value=int(spike_times[-1]), value=11845643)
    neural_end_sample = st.number_input('End sample for neural behavioral',
                                        min_value=neural_start_sample, max_value=int(spike_times[-1]), value=119845643)
    neural_start = spike_times.astype(float) - neural_start_sample
    neural_end = spike_times.astype(float) - neural_end_sample
    start_idx = np.where(neural_start >= 0)[0][0]
    end_idx = np.where(neural_end <= 0)[0][-1]
    spike_times = spike_times[start_idx:end_idx] - spike_times[start_idx]
    clusters = clusters[start_idx:end_idx]
    st.text('')
    st.markdown('Behavioral alignment')
    st.text('')
    framerate = st.number_input('Behavioral framerate', min_value=10, max_value=None, value=60)
    behavior_start = st.number_input('Alignment frame start for behavior',
                                     min_value=0, max_value=len(new_predictions[f_index]), value=28806)
    behavior_end = st.number_input('Alignment frame end for behavior',
                                   min_value=behavior_start, max_value=len(new_predictions[f_index]), value=244806)
    predictions = new_predictions[f_index][behavior_start:behavior_end]

if st.sidebar.checkbox('Find behaviors'):
    bout_frames = []
    term_frame = []
    group_start = [0]
    group_start = np.hstack((group_start, np.where(np.diff(predictions) != 0)[0]))
    behavior = predictions[group_start]
    behavior_duration = np.hstack((np.diff(group_start), len(predictions) - group_start[-1] + 1)) / framerate
    behavioral_start_time = group_start / framerate
    selected_b = st.number_input('Which behavior', min_value=0, max_value=len(np.unique(predictions)) - 1, value=0)
    if st.button('Start extracting latent variables for behavior {}'.format(selected_b)):
        spike_trains_all_behaviors = []
        for b in range(len(np.unique(predictions)) - 1):
            behavior_idx = np.where(behavior == b)[0]
            spike_trains = []
            # behavior_duration[behavior == b]
            for instance in range(len(behavior_idx)):
                spike_trains_trial_i = []
            # for instance in range(100):
            #     spike_trains_trial_i = []
                if behavior_duration[behavior == b][instance] > 0.1:
                    behavior_i = behavioral_start_time[behavior_idx[instance]]

                    for neuron in good_clusters:
                        # if instance == 0:
                        #     print(neuron)
                        neuron_spike = spike_times[clusters == neuron] / sample_rate
                        # print(neuron_spike, behavior_i, instance)
                        try:
                            start_idx = np.where(neuron_spike > (behavior_i - 1))[0][0]
                            end_idx = np.where(neuron_spike < (behavior_i + 2))[0][-1]
                            # print(start_idx, end_idx)
                            spike_times_s = [i for i in neuron_spike[start_idx:end_idx] - behavior_i]
                            if not spike_times_s:
                                spike_times_s = [np.array([np.nan])]
                            # else:
                                # spike_times_ss.append([np.nan])
                        except:
                            # print('except')
                            pass
                            # st.write('skipped neuron{}'.format(neuron))
                            # st.write('catch exception at instance {}, neuron {}'.format(instance, neuron))
                            # spike_times_s = [np.array([np.nan])]
                            # spike_times_ss.append(spike_times_s)
                        try:
                            spike_trains_trial_i.append(neo.SpikeTrain(np.concatenate(spike_times_s),
                                                                       units='sec', t_start=-1, t_stop=2))
                        except:
                            # spike_trains_trial_i = []
                        #     spike_trains_trial_i.append(neo.SpikeTrain(
                        #         np.array([np.nan]), units='sec', t_start=-1, t_stop=2))
                            pass
                        # except IndexError:
                        #     pass
                    # with open(os.path.join('/Users/ahsu/Downloads',
                    #                        '_spike_trains_i_{}_{}.sav'.format(instance, neuron)), 'wb') as f:
                    #     joblib.dump([spike_times_ss], f)
                if spike_trains_trial_i:
                    spike_trains.append(spike_trains_trial_i)
            with open(os.path.join('/Users/ahsu/Downloads', '_spike_trains_{}.sav'.format(b)), 'wb') as f:
                joblib.dump([spike_trains], f)
            spike_trains_all_behaviors.append(spike_trains)
        with open(os.path.join('/Users/ahsu/Downloads', '_spike_trains_all.sav'.format(b)), 'wb') as f:
            joblib.dump([spike_trains_all_behaviors], f)
            # bin_size = st.slider('bin size in milliseconds', min_value=5, max_value=50, value=20) * pq.ms
            # latent_dim = st.slider('number of latent dimensions', min_value=3, max_value=10, value=3)
            # gpfa_3dim = GPFA(bin_size=bin_size, x_dim=latent_dim)
            # trajectories = gpfa_3dim.fit_transform(spike_trains)
            # gpfa_dynamics = gpfa_dynamics_plots(bin_size, None, trajectories, None, None, None)
            # f4 = gpfa_dynamics.plot_state_space_3d()
            # st.pyplot(f4)

if st.sidebar.checkbox('plot rasters'):
    with open(os.path.join('/Users/ahsu/Downloads', '_spike_trains_10.sav'), 'rb') as fr:
        [spike_trains] = joblib.load(fr)
    trial_to_plot1 = st.number_input('Trial number', min_value=0, max_value=len(spike_trains), value=0)
    trial_to_plot2 = st.number_input('Trial number', min_value=0, max_value=len(spike_trains), value=5)
    trial_to_plot3 = st.number_input('Trial number', min_value=0, max_value=len(spike_trains), value=10)
    f1, ax1 = plt.subplots(1, 1, figsize=(5, 10))
    ax1.set_title(f'Raster plot of trial {trial_to_plot1}')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Neuron id')
    r = np.linspace(0, 1, len(np.unique(good_clusters)))
    c = plt.cm.tab20(r)
    for i, spike_train in enumerate(spike_trains[trial_to_plot1]):
        ax1.plot(spike_train, np.ones_like(spike_train) * i, ls='', marker='|', c=c[i])
        # ax1.axvline(x=0)
    plt.tight_layout()
    f2, ax1 = plt.subplots(1, 1, figsize=(5, 10))
    ax1.set_title(f'Raster plot of trial {trial_to_plot2}')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Neuron id')
    r = np.linspace(0, 1, len(np.unique(good_clusters)))
    c = plt.cm.tab20(r)
    for i, spike_train in enumerate(spike_trains[trial_to_plot2]):
        ax1.plot(spike_train, np.ones_like(spike_train) * i, ls='', marker='|', c=c[i])
        # ax1.axvline(x=0)
    plt.tight_layout()
    f3, ax1 = plt.subplots(1, 1, figsize=(5, 10))
    ax1.set_title(f'Raster plot of trial {trial_to_plot3}')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Neuron id')
    r = np.linspace(0, 1, len(np.unique(good_clusters)))
    c = plt.cm.tab20(r)
    for i, spike_train in enumerate(spike_trains[trial_to_plot3]):
        ax1.plot(spike_train, np.ones_like(spike_train) * i, ls='', marker='|', c=c[i])
        # ax1.axvline(x=0)
    plt.tight_layout()
    col1, col2, col3 = st.beta_columns([1, 1, 1])
    col1.pyplot(f1)
    col2.pyplot(f2)
    col3.pyplot(f3)

if st.sidebar.checkbox('Determine dimensions'):
    x_dims = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    log_likelihoods = []
    for x_dim in x_dims:
        gpfa_cv = GPFA(x_dim=x_dim)
        # estimate the log-likelihood for the given dimensionality as the mean of the log-likelihoods
        # from 3 cross-vailidation folds
        cv_log_likelihoods = cross_val_score(gpfa_cv, spike_trains, cv=3, n_jobs=3, verbose=True)
        log_likelihoods.append(np.mean(cv_log_likelihoods))
    with open(os.path.join('/Users/ahsu/Downloads', '_log_likelihoods.sav'), 'wb') as f:
        joblib.dump([log_likelihoods], f)
    f = plt.figure(figsize=(7, 5))
    plt.xlabel('Dimensionality of latent variables')
    plt.ylabel('Log-likelihood')
    plt.plot(x_dims, log_likelihoods, '.-')
    plt.plot(x_dims[np.argmax(log_likelihoods)], np.max(log_likelihoods), 'x', markersize=10, color='r')
    plt.tight_layout()
    st.pyplot(f)

if st.sidebar.checkbox('run gpfa'):
    with open(os.path.join('/Users/ahsu/Downloads', '_spike_trains_10.sav'), 'rb') as fr:
        [spike_trains] = joblib.load(fr)

    bin_size = st.slider('bin size in milliseconds', min_value=5, max_value=50, value=20) * pq.ms
    latent_dim = st.slider('number of latent dimensions', min_value=3, max_value=10, value=4)
    np.random.seed(0)
    spike_trains_training = []

    for i in range(len(spike_trains)):
        if np.random.rand() > 0.2:
            spike_trains_training.append(spike_trains[i])
    # st.write(spike_trains_training[0][0])
    np.random.seed(0)
    random.shuffle(spike_trains_training)
    # st.write(spike_trains_training[0][0])
    # spike_trains_training = np.random.choice(spike_trains, 200)
    np.random.seed(0)
    gpfa_3dim = GPFA(bin_size=bin_size, x_dim=latent_dim)
    if st.button('start'):
        gpfa_3dim.fit(spike_trains_training)
        trajectories = gpfa_3dim.transform(spike_trains[0:10])
        gpfa_dynamics = gpfa_dynamics_plots(bin_size, None, trajectories, None, None, None)
        f4 = gpfa_dynamics.plot_state_space_3d()
        # f4.suptitle('first half')
        st.pyplot(f4)
        # trajectories = gpfa_3dim.transform(spike_trains[len(spike_trains)//2:])
        # gpfa_dynamics = gpfa_dynamics_plots(bin_size, None, trajectories, None, None, None)
        # f5 = gpfa_dynamics.plot_state_space_3d()
        # f5.suptitle('second half')
        # st.pyplot(f5)

    # bin_size = st.slider('bin size in milliseconds', min_value=5, max_value=50, value=20) * pq.ms
    # latent_dim = st.slider('number of latent dimensions', min_value=3, max_value=10, value=3)
    # gpfa_3dim = GPFA(bin_size=bin_size, x_dim=latent_dim)
    # # st.write(len(spike_trains))
    # trajectories = gpfa_3dim.fit_transform(spike_trains)
    # gpfa_dynamics = gpfa_dynamics_plots(bin_size, None, trajectories, None, None, None)
    # f4 = gpfa_dynamics.plot_state_space_3d()
    # st.pyplot(f4)

    # for neuron in good_clusters:
    #     # st.write(neuron)
    #     neuron_spike = spike_times[clusters == neuron] / sample_rate
    #     # st.write(neuron_spike[0:50])
    #
    #     for b in range(len(np.unique(predictions))):
    #         behavior_idx = np.where(behavior == b)[0]
    #         spike_trains = []
    #         spike_trains_b = np.empty((len(behavior_idx), 3000))
    #         for instance in range(len(behavior_idx)):
    #             behavior_i = behavioral_start_time[behavior_idx[instance]]
    #             # st.write(behavior_i)
    #             # if instance == 0 and b == 0:
    #                 # st.write(neuron_spike)
    #             if b == 0 and instance <= 5:
    #
    #                 start_idx = np.where(neuron_spike > (behavior_i - 0.2))[0][0]
    #                 end_idx = np.where(neuron_spike < (behavior_i + 0.2))[0][-1]
    #                 spike_trains_trial_i.append(neuron_spike[start_idx:end_idx])
    #                 # st.write(neuron_spike > (behavior_i - 0.2)neuron_spike < (behavior_i + 0.2))
    #                 # spikes_i = neuron_spike[neuron_spike > (behavior_i - 0.2) and neuron_spike < (behavior_i + 0.2)] - behavior_i
    #                 # st.write(spikes_i)
    #                 # st.write(spikes_i.shape)
    #                 # st.write(spike_trains_b[instance, 0:len(spikes_i)].shape)
    #                 # spike_trains_b[instance, 0:len(spike_trains_trial_i[])] = np.reshape(spikes_i, (len(spikes_i), ))
    #                 # st.write(spike_trains_b)
    #                 # spike_train_i[instance, 1:len()]
    #             # if instance == 0:
    #             #     st.write(spikes_i)
    #         spike_trains.append(spike_trains_trial_i)
    # for i, group_num in enumerate(np.unique(predictions)):
    #     group_start.append(np.where(np.diff(predictions) != 1)[0])
    #     st.write(group_start)
    #     bout_frames.append(np.array(np.where(predictions == group_num)))
    # st.write(bout_frames[i][0:50])
    # term_f = np.diff(bout_frames[i]) != 1
    # st.write(term_f)
    # lengths, pos, grp = rle(term_frame.T)

if st.sidebar.checkbox('Find spiketrains that correlate with behavior', key='spiketrains'):
    neural_behavioral = bsoid_gpfa(spike_times, clusters, predictions)
    # neural_behavioral.main()

if st.sidebar.checkbox('Decode latent variables using GPFA', key='gpfa'):
    neural_behavioral = bsoid_gpfa(spike_times, clusters, predictions)
    neural_behavioral.main()

    # idx = []
    # for i, neuron in enumerate(np.unique(cluster_labels['cluster_id'])):
    #     if cluster_labels['group'][i] == 'good' or cluster_labels['group'][i] == 'mua':
    #         idx.append(np.where(clusters == neuron)[0])
    # r = np.linspace(0, 1, len(idx))
    # c = plt.cm.tab20(r)
    # f, ax1 = plt.subplots(1, 1, figsize=(15, 5))
    # spike_trains_trials = []
    # for _ in range(20):
    #     spike_trains = []
    #     for i in range(1, len(idx)):
    #         spike_times_s = spike_times[idx[i]] / 30000
    #         # spike_times[idx[0]]
    #         # st.write(np.concatenate(spike_times_s))
    #         # spike_trains.append(neo.AnalogSignal(spike_times_s, sampling_rate=1*pq.Hz, units='s'))
    #
    #         ax1.plot(spike_times_s, np.ones_like(spike_times_s) * i, ls='', marker='|', c=c[i])
    #         # plt.xlim(0, 60)
    #         # print(spike_times_s)
    #         try:
    #             spike_trains.append(neo.SpikeTrain(np.concatenate(spike_times_s), units='sec',
    #                                                t_stop=30))  # trial list, neuron list, spike times
    #         except:
    #             pass
    #     spike_trains_trials.append(spike_trains)
    # st.write(spike_trains_trials)
    # st.write(len(spike_trains_trials), len(spike_trains_trials[0]))
    # plt.tight_layout()
    # st.pyplot(f)
    # bin_size = st.slider('bin size in milliseconds', min_value=5, max_value=50, value=20) * pq.ms
    # latent_dim = st.slider('number of latent dimensions', min_value=3, max_value=10, value=3)
    # gpfa_3dim = GPFA(bin_size=bin_size, x_dim=latent_dim)
    # # st.write(spike_trains_trials[0][0])
    # trajectories = gpfa_3dim.fit_transform(spike_trains_trials)
    # gpfa_dynamics = gpfa_dynamics_plots(bin_size, None, trajectories, None, None, None)
    # f4 = gpfa_dynamics.plot_state_space_3d()
    # st.pyplot(f4)

# def gpfa(self):
#     gpfa_3dim = GPFA(bin_size=self.bin_size, x_dim=self.latent_dim)
#     fitting_all = st.radio('Portion to fit GPFA', options=('First 10%', 'First half', 'All'), index=2)
#     if st.button('Apply GPFA to simulated spike train', key='gpfa'):
#         if fitting_all == 'First 10%':
#             gpfa_3dim.fit(self.spiketrains_lorentz[:self.num_trials // 10])  # first half for training
#             trajectories = gpfa_3dim.transform(self.spiketrains_lorentz)
#         elif fitting_all == 'First half':
#             gpfa_3dim.fit(self.spiketrains_lorentz[:self.num_trials // 2])  # first half for training
#             trajectories = gpfa_3dim.transform(self.spiketrains_lorentz)
#         elif fitting_all == 'All':
#             trajectories = gpfa_3dim.fit_transform(self.spiketrains_lorentz)  # all


# def plot_simulated_rasters(spike_trains, num_clusters):
#     f, ax1 = plt.subplots(1, 1, figsize=(5, 10))
#     ax1.set_title(f'Raster plot')
#     ax1.set_xlabel('Time (s)')
#     ax1.set_ylabel('Neuron id')
#     r = np.linspace(0, 1, num_clusters)
#     c = plt.cm.tab20(r)
#     for i, spike_train in enumerate(spike_trains):
#         ax1.plot(spike_train, np.ones_like(spike_train) * i, ls='', marker='|', c=c[i])
#         plt.tight_layout()
