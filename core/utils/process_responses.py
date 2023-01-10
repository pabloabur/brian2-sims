from elephant.statistics import time_histogram, instantaneous_rate, kernels
from elephant.conversion import BinnedSpikeTrain
from brian2 import ms
import quantities as q
import numpy as np
import neo

def label_spikes(spikes, interval, bin_size):
    """ Process spikes upon presentation of multiple samples

    Args:
        spikes (neo.SpikeTrains): The spikes of a neuron
            during a single run, with samples one after the other.
        interval (numpy.array): Time stamps of initial and final
            times of one input class, represented by adjancents indexes
            e.g. interval[0] until interval[1], and so on.
        bin_size (quantity): bin size used to create histograms

    Returns:
        rates (neo.AnalogSignal): Time histograms, as rate, of each neuronal
            response.
    """
    # Converts into samples on each list entry
    spikes_on_samples = []
    for l in range(len(interval)-1):
        spikes_on_samples.append(
            spikes.time_slice(
                interval[l]*q.ms,
                interval[l+1]*q.ms).time_shift(-interval[l]*q.ms)
            )

    rates = []
    for spks in spikes_on_samples:
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

def monitor2binnedneo(monitor, time_interval, bin_size):
    """ Converts data from brian2 monitors to neo spikes. """
    neo_spks = []
    for spk_trains in monitor.spike_trains().values():
        neo_spks.append(neo.SpikeTrain(spk_trains/ms*q.ms,
                                       t_stop=time_interval*q.ms))

    return BinnedSpikeTrain(neo_spks, bin_size=bin_size*q.ms)

def neurons_rate(monitor, duration, sigma=200):
    spk_trains = monitor.spike_trains()
    spk_trains = [neo.SpikeTrain(spk_trains[x]/ms, t_stop=duration, units='ms')
                  for x in spk_trains]
    kernel = kernels.GaussianKernel(sigma=sigma*q.ms)
    rates = instantaneous_rate(spk_trains,
                               sampling_period=1*q.ms,
                               kernel=kernel)

    return rates
