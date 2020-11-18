import matplotlib.pyplot as plt
import numpy as np


class simulate_plots_2d:

    def __init__(self, oscillator_trajectory_2dim, num_spike_trains, oscillator_trajectory_ndim, times_oscillator,
                 spiketrains_oscillator, trial_to_plot):
        self.oscillator_trajectory_2dim = oscillator_trajectory_2dim
        self.num_spike_trains = num_spike_trains
        self.oscillator_trajectory_ndim = oscillator_trajectory_ndim
        self.times_oscillator = times_oscillator
        self.spiketrains_oscillator = spiketrains_oscillator
        self.trial_to_plot = trial_to_plot

    def plot_harmonics(self):
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        ax1.set_title('2-dim Harmonic Oscillator')
        ax1.set_xlabel('time [s]')
        c = ['black', 'hotpink']
        for i, y in enumerate(self.oscillator_trajectory_2dim):
            ax1.plot(self.times_oscillator, y, label=f'dimension {i}', c=c[i])
        ax1.legend()
        ax2.set_title('Trajectory in 2-dim space')
        ax2.set_xlabel('Dim 1')
        ax2.set_ylabel('Dim 2')
        ax2.set_aspect(1)
        ax2.plot(self.oscillator_trajectory_2dim[0], self.oscillator_trajectory_2dim[1], color='r')
        plt.tight_layout()
        return f

    def plot_high_dimensional(self):
        f, ax1 = plt.subplots(1, 1, figsize=(5, 10))
        ax1.set_title(f'Projection to {self.num_spike_trains}-dim space')
        ax1.set_xlabel('time [s]')
        y_offset = self.oscillator_trajectory_ndim.std() * 3.2
        r = np.linspace(0, 1, self.num_spike_trains)
        c = plt.cm.tab20(r)
        for i, y in enumerate(self.oscillator_trajectory_ndim):
            ax1.plot(self.times_oscillator, y + i * y_offset, c=c[i])
        plt.tight_layout()
        return f

    def plot_simulated_rasters(self):
        f, ax1 = plt.subplots(1, 1, figsize=(5, 10))
        ax1.set_title(f'Raster plot of trial {self.trial_to_plot}')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Neuron id')
        r = np.linspace(0, 1, self.num_spike_trains)
        c = plt.cm.tab20(r)
        for i, spike_train in enumerate(self.spiketrains_oscillator[self.trial_to_plot]):
            ax1.plot(spike_train, np.ones_like(spike_train) * i, ls='', marker='|', c=c[i])
        plt.tight_layout()
        return f


class gpfa_dynamics_plots:

    def __init__(self, bin_size, oscillator_trajectory_2dim, trajectories, trial_to_plot,
                 num_steps_transient, times_trial):
        self.bin_size = bin_size
        self.oscillator_trajectory_2dim = oscillator_trajectory_2dim
        self.trajectories = trajectories
        self.trial_to_plot = trial_to_plot
        self.num_steps_transient = num_steps_transient
        self.times_trial = times_trial

    def plot_state_space_2d(self):
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        ax1.set_title(f'Decoded trajectory for trial {self.trial_to_plot}')
        ax1.set_xlabel('Time [s]')
        times_trajectory = np.arange(len(self.trajectories[self.trial_to_plot][0])) * self.bin_size.rescale('s')
        ax1.plot(times_trajectory, self.trajectories[0][0], c='black', label="Dim 1")
        ax1.plot(times_trajectory, self.trajectories[0][1], c='hotpink', label="Dim 2")
        ax1.legend()
        ax2.set_title('Latent dynamics extracted by GPFA')
        ax2.set_xlabel('Dim 1')
        ax2.set_ylabel('Dim 2')
        ax2.set_aspect(1)
        # single trial trajectories
        for single_trial_trajectory in self.trajectories:
            ax2.plot(single_trial_trajectory[0], single_trial_trajectory[1], '-', lw=0.5, c='gray', alpha=0.5)
        # trial averaged trajectory
        average_trajectory = np.mean(self.trajectories, axis=0)
        ax2.plot(average_trajectory[0], average_trajectory[1], '-', lw=2, c='red', label='Trial averaged trajectory')
        ax2.legend()
        plt.tight_layout()
        return f

    def plot_state_space_3d(self):
        f = plt.figure(figsize=(15, 10))
        ax1 = f.add_subplot(2, 2, 1)
        ax2 = f.add_subplot(2, 2, 2, projection='3d')
        c = ['hotpink', 'deepskyblue', 'black']
        # single trial trajectories
        ax2.set_title('Latent dynamics extracted by GPFA')
        ax2.set_xlabel('Dim 1')
        ax2.set_ylabel('Dim 2')
        ax2.set_zlabel('Dim 3')
        for single_trial_trajectory in self.trajectories:
            ax2.plot(single_trial_trajectory[0], single_trial_trajectory[1], single_trial_trajectory[2],
                     lw=0.5, c='gray', alpha=0.5)
        # trial averaged trajectory
        average_trajectory = np.mean(self.trajectories, axis=0)
        ax2.plot(average_trajectory[0], average_trajectory[1], average_trajectory[2],
                 lw=2, c='red', label='Trial averaged trajectory')
        ax2.legend()
        ax1.set_title('Decoded trial average trajectory')
        ax1.set_xlabel('Time [s]')
        for i, x in enumerate(average_trajectory):
            ax1.plot(np.arange(len(x)) * 0.02, x, label=f'Dim {i + 1}', c=c[i])
        ax1.legend()
        ax2.view_init(azim=-5, elev=75)
        plt.tight_layout()
        return f


class simulate_plots_3d:

    def __init__(self, times, lorentz_trajectory_3dim, transient_duration, num_steps_transient):
        self.times = times
        self.lorentz_trajectory_3dim = lorentz_trajectory_3dim
        self.transient_duration = transient_duration
        self.num_steps_transient = num_steps_transient

    def plot_lorentz(self):
        f = plt.figure(figsize=(15, 10))
        ax1 = f.add_subplot(2, 2, 1)
        ax2 = f.add_subplot(2, 2, 2, projection='3d')
        c = ['black', 'hotpink', 'deepskyblue']
        ax1.set_title('Lorentz system')
        ax1.set_xlabel('Time [s]')
        labels = ['x', 'y', 'z']
        for i, x in enumerate(self.lorentz_trajectory_3dim):
            ax1.plot(self.times, x, label=labels[i], c=c[i])
        ax1.axvspan(-self.transient_duration.rescale('s').magnitude, 0, color='gray', alpha=0.1)
        ax1.text(-5, -20, 'Initial transient', ha='center')
        ax1.legend()
        ax2.set_title(f'Trajectory in 3-dim space')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_ylabel('z')
        ax2.plot(self.lorentz_trajectory_3dim[0, :self.num_steps_transient],
                 self.lorentz_trajectory_3dim[1, :self.num_steps_transient],
                 self.lorentz_trajectory_3dim[2, :self.num_steps_transient], c='gray', alpha=0.3)
        ax2.plot(self.lorentz_trajectory_3dim[0, self.num_steps_transient:],
                 self.lorentz_trajectory_3dim[1, self.num_steps_transient:],
                 self.lorentz_trajectory_3dim[2, self.num_steps_transient:], c='red')
        return f
