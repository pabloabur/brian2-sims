from elephant.statistics import time_histogram, instantaneous_rate, kernels
from elephant.conversion import BinnedSpikeTrain
import pandas as pd
from brian2 import ms
import quantities as q
import numpy as np
import neo

def objects2dataframe(objects, object_variables):
    """ Saves brian objects in tidy format.
    
    Parameters
    ----------
    objects : list of brian objects
    object_variables : list of tuples
        Every tuple `i` corresponds to the variables of object `i` that will
        be saved

    Returns
    -------
    output_df : pandas.Dataframe
    """
    df_dict = {}
    for obj, vars in zip(objects, object_variables):
        for var in vars:
            df_dict[var] = np.array(getattr(obj, var))
    output_df = pd.DataFrame(df_dict)
    return output_df

def statemonitors2dataframe(monitors):
    """ Saves data from brian.StateMonitor in tidy format.
    
    Parameters
    ----------
    monitors : list of brian.StateMonitor

    Returns
    -------
    output_df : pandas.Dataframe
    """
    val_col, t_col, mon_col, var_col, id_col = [], [], [], [], []
    for mon in monitors:
        variables = mon.needed_variables
        data_packet = mon.get_states(variables, units=False)
        for var in variables:
            temp_val = data_packet[var]
            temp_t = np.tile(mon.t/ms, temp_val.shape[1])
            temp_id = np.repeat([mon.record[x] for x in range(mon.n_indices)], temp_val.shape[0])
            temp_var = np.tile(var, temp_val.size)
            temp_mon = [mon.name for _ in range(temp_val.size)]

            val_col.extend(temp_val.flatten(order='F'))
            t_col.extend(temp_t)
            mon_col.extend(temp_mon)
            var_col.extend(temp_var)
            id_col.extend(temp_id)

    output_df = pd.DataFrame(
        {'value': val_col,
         'time_ms': t_col,
         'monitor': mon_col,
         'variable': var_col,
         'id': id_col})

    return output_df

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
