import sys
import warnings

from brian2 import mV, ms, mA, second, Hz, TimedArray, check_units, run,\
         SpikeGeneratorGroup, SpikeMonitor, Function, DEFAULT_FUNCTIONS

from scipy.stats import pearsonr, spearmanr
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

import re

from core.utils.misc import minifloat2decimal, decimal2minifloat

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from random import randint
from pathlib import Path
import pickle

def replicate_sequence(num_channels, reference_indices, reference_times,
                       sequence_duration, duration):
    # Replicates sequence throughout simulation
    input_spikes = SpikeGeneratorGroup(num_channels, reference_indices,
                                       reference_times,
                                       period=sequence_duration*ms)
    input_monitor = SpikeMonitor(input_spikes)
    print('Generating input...')
    run(duration*ms)
    spike_indices = np.array(input_monitor.i)
    spike_times = np.array(input_monitor.t/ms)

    return spike_indices, spike_times

def neuron_rate(spike_source, kernel_len, kernel_var, simulation_dt,
                interval=None, smooth=False, trials=1):
    """Computes firing rates of neurons in a SpikeMonitor. DEPRECATED

    Args:
        spike_source (brian2.SpikeMonitor): Source with spikes and times. It
            can be a monitor or a dictionary with {'i': [i1, i2, ...],
            't': [t1, t2, ...]}
        kernel_len (Brian2.unit): Length of the averaging kernel in units of
            time.
        kernel_var (Brian2.unit): Variance of the averaging kernel in units of
            time.
        simulation_dt (Brian2.unit): Time scale of simulation's time step
        interval (list of int): lower and upper values of the interval, in
            Brian2 units of time, over which the rate will be calculated.
            If None, the whole recording provided is used.
        smooth (boolean): Flag to indicate whether rate should be calculated
            with a smoothing gaussian window.
        trials (int): Number of trials over which result will be averaged.

    Returns:
        neuron_rates (dict): Rates (in Hz) and corresponding instants of each
            neuron. Rate values are stored in a matrix with each line
            corresponding to rates of a given neuron over time.
    """
    # Generate inputs
    if isinstance(spike_source, SpikeMonitor):
        spike_trains = spike_source.spike_trains()
    elif isinstance(spike_source, dict):
        # Convert to monitor so spike_trains() can be called
        num_indices = max(spike_source['i']) + 1
        spike_gen = SpikeGeneratorGroup(num_indices,
                                           spike_source['i'],
                                           spike_source['t'])
        spike_mon = SpikeMonitor(spike_gen)
        run(max(spike_source['t']))
        spike_trains = spike_mon.spike_trains()
    else:
        print('Spiking source was not recognized.')
        sys.exit()

    # Convert objects for convenience
    samples_in_bin = np.around(kernel_len/simulation_dt).astype(int)
    spike_trains = [np.around(val/simulation_dt).astype(int)
        for val in spike_trains.values()]

    # Defines general intervals and bins. Bins without spikes are
    # ignored to speed up calculations
    if interval:
        min_sample = np.around(interval[0]/simulation_dt).astype(int)
        max_sample = np.around(interval[1]/simulation_dt).astype(int)
    else:
        min_sample = np.nanmin([min(x) if x.any() else np.nan for x in spike_trains])
        if min_sample < samples_in_bin:
            min_sample = 0
        else:
            # min_sample is adjusted according to samples_in_bin to
            # consider minimum spike time as starting point for calculations
            min_sample -= min_sample%samples_in_bin
        max_sample = np.nanmax([max(x) if x.any() else np.nan for x in spike_trains])
    intervals = range(int(min_sample), int(max_sample))
    if len(intervals) % trials:
        warnings.warn(f'Trials must divide interval in even parts. Using '
                      f' one trial for now...')
        trials = 1

    # Creates regular bins
    intervals = np.array_split(intervals, trials)
    n_bins = (intervals[0][-1]-intervals[0][0]) // samples_in_bin
    if not n_bins:
        # Ensure at least one bin is processed
        n_bins = 1
    bins_length = [samples_in_bin for _ in range(n_bins)]

    # Creates last irregular bin and update histogram interals
    last_bin_length = (intervals[0][-1]-intervals[0][0]) % samples_in_bin
    if last_bin_length:
        bins_length.append(last_bin_length)
        n_bins += 1

    rates = np.zeros((np.shape(spike_trains)[0], n_bins, trials))
    spike_times = np.zeros(n_bins)

    for trial in range(trials):
        # TODO vectorize histogram calculation
        for idx, neu_spike_times in enumerate(spike_trains):
            bins = ([intervals[trial][0]]
                  + [intervals[trial][0]+x for x in np.cumsum(bins_length)])
            # Use histogram to get values that will be convolved
            h, b = np.histogram(neu_spike_times,
                bins=bins,
                range=intervals[trial], density=False)
            rates[idx, :, trial] = h/(bins_length*simulation_dt)
    rates = np.sum(rates, axis=2)/trials
    if trials > 1:
        # Use last trial to make spike times start at 0
        spike_times = np.array(bins[:-1]) - bins[0]
    else:
        spike_times = b[1:]

    # Consider values for each simulation dt
    interp_func = interp1d(spike_times, rates, kind='next')
    spike_times = np.arange(min(spike_times), max(spike_times), 1)

    neuron_rates = {}
    rates = interp_func(spike_times)

    if smooth:
        # Create normalized and truncated gaussian time window
        kernel_var /= simulation_dt
        neuron_rates['smoothed'] = gaussian_filter1d(rates,
                                                     kernel_var,
                                                     output=float)*Hz
        # Alternatively use numpy.convolve with normalized window
        #kernel_limit = np.floor(samples_in_bin/2)
        #lower_limit = -kernel_limit
        #upper_limit = kernel_limit + 1 if samples_in_bin % 2 else kernel_limit
        #kernel = np.exp(-(np.arange(lower_limit, upper_limit)) ** 2 / (2 * kernel_var ** 2))
        #kernel = kernel[np.where(kernel>.001)]
        #kernel = kernel / kernel.sum()

        #neuron_rates['smoothed'] = np.convolve(
        #    newrate, kernel, mode='same')
        
    neuron_rates['rate'] = rates*Hz
    neuron_rates['t'] = spike_times*simulation_dt

    return neuron_rates

def label_ensembles(sequence, neurons_rate, rate_thr=10*Hz):
    """ Generate a connectivity pattern that can be used to label 
        neurons according to rate activity

        Args:
            items (iterable): Contains integers that represent items. It must 
                have sequential order.
            neurons_rate (np.array): Two dimensional matrix where each row 
                contains rate of a neuron over time.
            rate_thr (int): Threshold that defines if neuron will be labeled 
                as representing given item.

        Returns:
            conn_pattern (dict of list): Source and target indices that are 
            to be used as i and j paramters of Brian2 connect method.
    """
    item_len = sequence.item_length
    rates = neurons_rate['rate']
    conn_pattern = {'source': [], 'target': []}

    for neu in range(np.shape(rates)[0]):
        for item in sequence.items.keys():
            if np.any(rates[neu, item*item_len:(item+1)*item_len] > rate_thr):
                conn_pattern['source'].append(neu)
                conn_pattern['target'].append(item)

    return conn_pattern

def random_integers(a, b, _vectorization_idx):
    random_samples = []
    for sample in range(len(_vectorization_idx)):
        random_samples.append(randint(a, b))

    return np.array(random_samples)
random_integers = Function(random_integers, arg_units=[1, 1], return_unit=1,
                           stateless=False, auto_vectorise=True)

def permutation_from_rate(neurons_rate):
    """This functions uses the instant of maximum firing rate to extract
    permutation indices that can be used to sort a raster plot so that
    an activity trace (relative to a given task) is observed.

    Args:
        neurons_rate (dict): Dictionary with firing rate values for each
            neuron. Keys must be neuron index and 'rate' or 't'. Rate values
            is stored in matrix with each line representing rate of a given
            neuron over time

    Returns:
        permutation_ids (list): Permutation indices.
    """
    num_neu = np.shape(neurons_rate['rate'])[0]
    num_samples = np.shape(neurons_rate['rate'])[1]

    average_rates = np.zeros((num_neu, num_samples))*np.nan
    # Proxy time reference
    temp_t = np.array([x for x in range(num_samples)])
    peak_instants = {}

    for neu_id, neu_act in enumerate(neurons_rate['smoothed']):
        average_rates[neu_id, :] = neu_act

        # Consider only spiking neurons
        if average_rates[neu_id].any():
            # Get first peak found on rate
            peak_index = [np.argmax(average_rates[neu_id])]
            peak_instants[neu_id] = temp_t[peak_index]

    # Add unresponsive neurons again
    sorted_peaks = dict(sorted(peak_instants.items(), key=lambda x: x[1]))
    permutation_ids = [x[0] for x in sorted_peaks.items()]
    [permutation_ids.append(neu) for neu in range(num_neu) if not neu in permutation_ids]

    return permutation_ids

def load_merge_multiple(path_name, file_name, mode='pickle', allow_pickle=False):
    merged_dict = {}
    if mode == 'pickle':
        for pickled_file in Path(path_name).glob(file_name):
            with open(pickled_file, 'rb') as f:
                for key, val in pickle.load(f).items():
                    try:
                        merged_dict.setdefault(key, []).extend(val)
                    except TypeError:
                        merged_dict.setdefault(key, []).append(val)
    elif mode == 'numpy':
        for saved_file in sorted(Path(path_name).glob(file_name),
                                 key=lambda path: int(path.stem.split('_')[1])):
            data = np.load(saved_file, allow_pickle=allow_pickle)
            for key, val in data.items():
                if key == 'rec_ids' or key == 'ff_ids' or key == 'ffi_ids':
                    # These connections do not change. There could be
                    # an if conditional here but that would slow things down
                    merged_dict[key] = val
                else:
                    try:
                        merged_dict[key] = np.hstack((merged_dict[key], val))
                    except KeyError:
                        merged_dict[key] = val

    return merged_dict

# TODO teili for functions used below
#def bar_from_recording(filename):
#    """ Converts aedat output with events and save arrays to disk
#    
#    Args:
#        filename (str): Name of the file containing events, e.g.
#            'dvSave-7V_normal_fullrec.aedat4'
#    """
#    print('converting events...')
#    ev_npy = aedat2numpy(filename, version='V4')
#
#    print('getting time and indices...')
#    i_on, t_on, i_off, t_off = dvs2ind(ev_npy, resolution=(346, 260))
#
#    print('saving on disk...')
#    np.savez(filename+'_events', on_indices=i_on, on_times=t_on, off_indices=i_off, off_times=t_off)

def recorded_bar_testbench(filename, num_samples, repetitions):
    events = np.load(filename)
    ref_input_indices = events['off_indices']
    ref_input_times = events['off_times']*ms
    ref_input_indices -= min(ref_input_indices)

    # Sampling input space
    #import matplotlib.pyplot as plt
    #plt.figure()
    #plt.plot(ref_input_times/ms, ref_input_indices, '.')
    rng = np.random.default_rng(12345)
    sampled_indices = rng.choice(np.unique(ref_input_indices), num_samples, replace=False)
    sampled_indices = np.in1d(ref_input_indices, sampled_indices)
    ref_input_times = ref_input_times[sampled_indices]
    ref_input_indices = ref_input_indices[sampled_indices]

    # Adjusting input size and removing gaps
    tmp_ind = np.unique(ref_input_indices)
    indices_mapping = {val: ind for ind, val in enumerate(tmp_ind)}
    ref_input_indices = np.vectorize(indices_mapping.get)(ref_input_indices)
    #plt.figure()
    #plt.plot(ref_input_times/ms, ref_input_indices, '.')
    #plt.figure()
    #a=np.unique(ref_input_indices)
    #plt.plot(np.arange(len(a)), a)
    #plt.show()

    # Repeat presentation
    input_times = ref_input_times
    input_indices = ref_input_indices
    recording_gap = 1*second
    for rep in range(repetitions):
        recording_duration = np.max(input_times)
        temp_times = ref_input_times + recording_duration + recording_gap
        input_times = np.concatenate((input_times, temp_times)) * second
        input_indices = np.concatenate((input_indices, ref_input_indices))

    return input_times, input_indices

def get_metrics(spike_monitor):
    data = pd.DataFrame({'i': np.array(spike_monitor.i),
                         't': np.array(spike_monitor.t/ms)})

    spike_indices, spike_times = data.sort_values(['i', 't']).values.T

    neu_ids, id_slices = np.unique(spike_indices, True)
    t_arrays = np.split(spike_times, id_slices[1:])
    max_id = int(max(neu_ids))

    isi = [np.diff(x) for x in t_arrays]
    cv = [np.std(x)/np.mean(x) if len(x)>1 else np.nan for x in isi]

    return isi, cv

def plot_norm_act(monitor):
    plt.figure()
    plt.plot(monitor.normalized_activity_proxy.T)
    plt.xlabel('time (ms)')
    plt.ylabel('normalized activity value')
    plt.title('Normalized activity of all neurons')
    plt.show()

def run_batches(Net, orca, training_blocks, training_duration,
                simulation_dt, path, monitor_params):
    """ This function can be used to divide the number of simulations
        in blocks, so it is a more friendly approach when simulating
        in a machine with low memory and computation resources.
    """
    remainder_time = int(np.around(training_duration/simulation_dt)
                         % training_blocks) * simulation_dt
    for block in range(training_blocks):
        block_duration = int(np.around(training_duration/simulation_dt)
                             / training_blocks) * simulation_dt
        Net.run(block_duration)
        # Free up memory
        orca.save_data(monitor_params, path, block)

        # Reinitialization. Remove first so obj reference is not lost
        Net.remove([x for x in orca.monitors.values()])
        orca.monitors = {}
        orca.create_monitors(monitor_params)
        Net.add([x for x in orca.monitors.values()])

    if remainder_time != 0:
        block += 1
        Net.run(remainder_time, report='stdout', report_period=100*ms)
        orca.save_data(monitor_params, path, block)
        Net.remove([x for x in orca.monitors.values()])
        orca.monitors = {}
        orca.create_monitors(monitor_params)
        Net.add([x for x in orca.monitors.values()])

def expand_state_variables(model, initial_name, source_group):
    """ This function takes a model generated by SynapseEquationBuilder
        and substitute specified state variables with an expanded names
        that can be understood by Brian2 e.g. sum_w (summed) can be
        replaced by sum_w_ff and sum_w_rec.

        Args:
            model (dict): Contains model descriptions.
            initial_name (str): Pattern to be replaced.
            source_group (str): Source group.
    """
    pattern = re.compile(initial_name)
    saved_index = None
    aditional_name = 'rec' if source_group=='pyr' else 'ff'
    # Keys in model description that will be processed
    keys = ['model', 'on_pre', 'on_post']
    for k in keys:
        temp_model = model.keywords[k].split('\n')
        for idx, line in enumerate(temp_model):
            if pattern.search(line) is not None:
                temp_model[idx] = line.replace(initial_name,
                    initial_name+aditional_name)
        model.keywords[k] = '\n'.join(temp_model)
        model.keywords_original[k] = '\n'.join(temp_model)

    return model

def plot_fp8():
    int_vals = [x for x in range(256)]
    dec_vals = minifloat2decimal(int_vals)

    # For simplicity, set -0, that is 128, to 0
    int_vals[128] = 0
    dec_vals.sort()
    plt.plot(dec_vals, np.zeros_like(dec_vals), '|')
    plt.show()
