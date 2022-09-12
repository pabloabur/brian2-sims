import sys
from itertools import chain
from pathlib import Path
import pickle
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import OrderedDict

import neo
import quantities as q
from elephant.statistics import time_histogram
from elephant.spike_train_generation import homogeneous_poisson_process
from viziphant.statistics import plot_time_histogram

def process_responses(spikes, interval, bin_size):
    """ Process multiple trials of spikes

    Args:
        spikes (neo.SpikeTrains): The spikes of a neuron
            during a single run, with multiple trials.
        interval (numpy.array): Time stamps of initial and final
            times of one input class, represented by adjancents indexes
            e.g. interval[0] until interval[1], and so on.
        bin_size (quantity): bin size of the histogram. Must be equal to 
            total duration of symbols

    Returns:
        rates (neo.AnalogSignal): Time histograms, as rate, of each neuronal
            trial.
    """
    # Converts into trials on each list entry
    spikes_on_trials = []
    for l in range(len(interval)-1):
        spikes_on_trials.append(
            spikes.time_slice(
                interval[l]*q.ms,
                interval[l+1]*q.ms).time_shift(-interval[l]*q.ms)
            )

    rates = []
    for spks in spikes_on_trials:
        aux_rates = time_histogram(spks, bin_size=bin_size,
                                   output='rate').magnitude.flatten()
        aux_rates[np.isnan(aux_rates)] = 0
        rates.append(aux_rates)

    return rates

def create_response_matrix(rates, n_threshold):
    """ Process responses from spiking activity.

    Args:
        rates (list of neo.AnalogSignal): Time histograms of a single neuron.
            Each entry holds the rates of a trial.
        n_threshold (int): Number of thresholds used to compute responses as
            in Quiroga et al. (2007).

    Returns:
        response (nd.array): Increasing index of the list represents increasing
            thresholds.
    """
    response = []
    fmin, fmax = np.min(rates), np.max(rates)
    for n_thr in range(n_threshold):
        thres = fmin + n_thr*(fmax - fmin)/n_threshold
        response.append(np.average(1*np.greater(rates, thres), axis=0))

    return np.transpose(response)


# Preparing matrices according to data loaded
data_folder = Path(sys.argv[1])
experiments = sorted(data_folder.glob('**/metadata.json'))
if not experiments:
    experiments = ['']

gaps, trials = [], []
for exp in experiments:
    with open(exp, 'r') as f:
        desc = json.load(f)
        gaps.append(desc['time_gap'])
        trials.append(desc['trial_no'])
gaps = list(OrderedDict.fromkeys(gaps))
trials = list(OrderedDict.fromkeys(trials))

train_metric = np.zeros((len(gaps), len(trials)))
train_metric.fill(np.nan)
test_metric = np.zeros((len(gaps), len(trials)))
test_metric.fill(np.nan)
# TODO 6 below hardcoded
test_selectivity = np.zeros((6, len(trials), len(gaps)))
test_selectivity.fill(np.nan)
# the worst case scenario
rand_metric = np.zeros((len(gaps), len(trials)))
rand_metric.fill(np.nan)
rand_selectivity = np.zeros((6, len(trials), len(gaps)))
rand_selectivity.fill(np.nan)
#I = np.roll(np.eye(9), axis=(0, 1), shift=(0, 1))  # in case it is slightly shifted
I = np.eye(9)
test_mat_large = None
test_mat_0 = None

# Iterate over each experiment, storing results in a matrix that is average later on
experiments = sorted(data_folder.glob('*'))

# experiments with multiple trials are stored in folders, but this script can
# also plot metrics for a single trial
removed_folders = []
for exp in experiments:
    if exp.is_dir():
        if len(list(exp.glob('*'))) <= 1:
            removed_folders.append(exp)
experiments = [exp for exp in experiments if exp.name != 'description.txt']
for rm in removed_folders: experiments.remove(rm)

# seq_presentations is the number of times sequence was presented to the
# network. Do not confuse with trial, which contains another instance of
# network.
seq_presentations = 10
for exp in experiments:
    if exp.is_file():
        exp = exp.parent
    with open(exp / 'metadata.json', 'r') as f:
        metadata = json.load(f)

    t_gap = metadata['time_gap']
    t_gap = gaps.index(t_gap)
    trial = metadata['trial_no']
    trial = trials.index(trial)
    seq_dur = float(metadata['sequence_duration'].split(' ')[0])
    num_channels = metadata['num_channels']
    num_items = metadata['num_items']
    bin_size = seq_dur*q.ms / num_items
    channel_groups = int(num_channels / num_items)
    sim_duration = float(metadata['sim_duration'].split(' ')[0])
    testing_duration = float(metadata['testing_duration'].split(' ')[0])
    interval_train = np.arange(sim_duration-testing_duration-seq_presentations*seq_dur,
                               sim_duration-testing_duration+seq_dur,
                               seq_dur)
    interval_test = np.arange(sim_duration-testing_duration,
                              sim_duration-testing_duration+(1+seq_presentations)*seq_dur,
                              seq_dur)

    with open(exp / 'input_spikes', 'rb') as f:
        input_spikes = pickle.load(f)
    with open(exp / 'output_spikes', 'rb') as f:
        output_spikes = pickle.load(f)

    # Some channels might not be active
    input_spikes = [i_spk for i_spk in input_spikes if np.any(i_spk)]
    input_spikes = [input_spikes[idx] for idx in range(0, len(input_spikes), channel_groups)]
    output_spikes = [o_spk if np.any(o_spk) else [0] for o_spk in output_spikes]
    max_out = max([max(x, default=0) for x in output_spikes], default=0)
    max_in = max([max(x, default=0) for x in input_spikes], default=0)
    tf = max([max_in, max_out], default=0)
    edges = (0, tf)
    input_spikes = [neo.SpikeTrain(spks, units='ms', t_stop=sim_duration) for spks in input_spikes]
    output_spikes = [neo.SpikeTrain(spks, units='ms', t_stop=sim_duration) for spks in output_spikes]

    n_threshold = 100
    train_resp_mat = np.zeros((len(output_spikes), num_items, n_threshold))
    train_resp_mat.fill(None)
    test_resp_mat = np.zeros((len(output_spikes), num_items, n_threshold))
    test_resp_mat.fill(None)
    rand_resp_mat = np.zeros((len(output_spikes), num_items, n_threshold))
    rand_resp_mat.fill(None)
    for row, spk_o in enumerate(output_spikes):
        rates = process_responses(spk_o, interval_train, bin_size)
        train_resp_mat[row, :, :] = create_response_matrix(rates, n_threshold)

        rates = process_responses(spk_o, interval_test, bin_size)
        test_resp_mat[row, :, :] = create_response_matrix(rates, n_threshold)

        rates = [time_histogram(
                    homogeneous_poisson_process(rate=50*q.Hz,
                                                t_start=0*q.ms,
                                                t_stop=seq_dur*q.ms),
                    bin_size=bin_size, output='rate'
                 ).magnitude.flatten() for x in range(num_items)]
        rand_resp_mat[row, :, :] = create_response_matrix(rates, n_threshold)

    train_metric[t_gap, trial] = np.mean(np.diag(train_resp_mat[:, :, 0]))
    test_metric[t_gap, trial] = np.mean(np.diag(test_resp_mat[:, :, 0]))
    test_selectivity[:, trial, t_gap] = 1 - 2*np.mean(np.mean(train_resp_mat,
                                                              axis=1),
                                                      axis=1)
    rand_metric[t_gap, trial] = np.mean(np.diag(rand_resp_mat[:, :, 0]))
    rand_selectivity[:, trial, t_gap] = 1 - 2*np.mean(np.mean(rand_resp_mat,
                                                              axis=1),
                                                      axis=1)
    # Saves data for plotting purposes
    if gaps[t_gap]==0 and trial==0:
        test_mat_0 = test_resp_mat
    elif gaps[t_gap]==39 and trial==0:
        test_mat_large = test_resp_mat

# Getting an idea of the metric given some spike trains
fig, axs = plt.subplots(2, 1, sharex=True)
for i, spk in enumerate(input_spikes):
    axs[0].plot(spk, [i]*len(spk), 'k.')

for i, spk in enumerate(output_spikes):
    axs[1].plot(spk, [i]*len(spk), 'k.')
plt.xlim([interval_train[0], interval_test[-1]])
plt.vlines(interval_train, ymin=-1, ymax=10)

plt.figure()
plt.imshow(train_resp_mat[:, :, 0], interpolation=None)
plt.title('response matrix - training')
plt.xlabel('input spike trains')
plt.ylabel('output spike trains')
plt.colorbar()

if np.any(test_mat_large):
    plt.figure()
    plt.imshow(test_mat_large[:, :, 0], interpolation=None)
    plt.title('synchrony coefficients - testing, large gaps')
    plt.xlabel('output spike trains')
    plt.ylabel('input spike trains')
    plt.colorbar()

plt.figure()
plt.imshow(rand_resp_mat[:, :, 0], interpolation=None)
plt.title('synchrony coefficients - poisson spikes')
plt.xlabel('output spike trains')
plt.ylabel('input spike trains')
plt.colorbar()

if np.any(test_mat_0):
    plt.figure()
    plt.imshow(test_mat_0[:, :, 0], interpolation=None)
    plt.title('confusion matrix') #testing, large gaps
    plt.xlabel('output labels')
    plt.ylabel('input labels')
    plt.colorbar()

plt.figure()
e = np.std(train_metric, axis=1) / np.sqrt(len(trials))
train_metric = np.average(train_metric, axis=1)
plt.errorbar(gaps, train_metric, e,
             label='training', linestyle='None', marker='^')
plt.legend()

e = np.std(test_metric, axis=1) / np.sqrt(len(trials))
test_metric = np.average(test_metric, axis=1)
plt.errorbar(gaps, test_metric, e,
             label='testing', linestyle='None', marker='v')
plt.legend()
plt.xlabel('Time delay between sequence presentation [ms]')
plt.ylabel('mean diagonal')
plt.ylim([-0.1, 1])
plt.grid()

plt.figure()
_ = plt.hist(np.mean(test_selectivity[:, :, 0], axis=1), label='readout spikes, no gaps')
_ = plt.hist(np.mean(rand_selectivity[:, :, 0], axis=1), label='random spikes, no gaps')
plt.xlabel('selectivity')
plt.ylabel('count')
plt.xlim([-0.1, 1.1])
plt.legend()

plt.show()

# TODO clustering. Objects=neurons, properties=response to items
#from scipy.cluster import hierarchy
#data = np.random.randn(100, 2)
#dist = hierarchy.distance.pdist(data, metric='euclidean')
## link[x] shows which items were merged to a cluster, distance, and size of it
#link = hierarchy.linkage(dist, 'single')
## you can search for indices in dist matrix to be sure
##ind = sch.fcluster(link, 0.5*d.max(), 'distance')? it flattens "dendogram"
## Use max_d to control number of clusters
#dend = hierarchy.dendrogram(link)
#index = dend['leaves']
#plt.imshow(data[:, index][index, :])
