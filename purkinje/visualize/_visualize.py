import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np

def plot_raster(raster, CS_onsets, CS_offsets, US_onsets, probe_start, probe_per, save_path):
    fig, ax = plt.subplots(figsize=(10,7.5))
    for i in range(raster.shape[0]):
        if i > probe_start and ((i - probe_start) % probe_per) == 0:
            color = 'r'
        else:
            color = 'k'
        trial = raster[i,...]
        spike_times = np.nonzero(trial)
        plt.scatter(spike_times, i*np.ones_like(spike_times), color=color, s=2)

    for s, stim in enumerate([CS_onsets, CS_offsets, US_onsets]):
        color = 'g' if s < 2 else 'r'
        for timing in stim:
            ax.axvline(x=(timing * 1000), ymin=0, ymax=len(raster), color=color, linewidth=4)
    #ax.set_xlim([-5,1000])
    ax.set_ylim([0,len(raster)])
    ax.set_ylabel('Trials', fontsize=32)
    ax.set_xlabel('Time (ms)', fontsize=32)
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.tick_params(axis='both', which='minor', labelsize=24)
    plt.savefig(save_path)

def mp_to_fr(membrane_potential, spike_potential, bin_width=5):

    membrane_potential = np.array(membrane_potential)
    spikes = 1*(membrane_potential == spike_potential)
    return 1000*np.convolve(spikes, np.ones(bin_width)/bin_width, mode='valid')

def plot_avg_fr(membrane_potentials, spike_potential, CS_onsets, CS_offsets, US_onsets, save_path, bin_width=5):
    
    fig, ax = plt.subplots(figsize=(10,7.5))

    all_fr = np.array([mp_to_fr(mp, spike_potential, bin_width=bin_width) for mp in membrane_potentials])

    mean_fr = all_fr.mean(0)
    err_fr  = all_fr.std(0)

    x = np.arange(0, len(mean_fr))

    ax.plot(x, all_fr.mean(0), color='b')
    ax.fill_between(x, mean_fr - err_fr, mean_fr + err_fr, color='b', alpha=.2)

    for s, stim in enumerate([CS_onsets, CS_offsets, US_onsets]):
        color = 'g' if s < 2 else 'r'
        for timing in stim:
            ax.axvline(x=(timing * 1000), color=color, linewidth=4)

    ax.set_ylabel('Firing rate (Hz)', fontsize=32)
    ax.set_xlabel('Time (ms)', fontsize=32)
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.tick_params(axis='both', which='minor', labelsize=24)

    plt.savefig(save_path)
