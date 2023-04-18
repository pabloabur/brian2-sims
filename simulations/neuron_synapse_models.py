#!/usr/bin/env python3
import numpy as np
from brian2 import mV, ms, Hz, SpikeMonitor, StateMonitor, TimedArray,\
    defaultclock, run, PoissonGroup, device

import pandas as pd
from scipy.stats import pearsonr
import json
import feather

from brian2tools import plot_state, brian_plot, plot_rate
import matplotlib.pyplot as plt
from elephant.spike_train_correlation import cross_correlation_histogram
import quantities as q

from core.utils.misc import minifloat2decimal
from core.utils.process_responses import neurons_rate
from core.utils.process_responses import monitor2binnedneo

from core.equations.neurons.LIF import LIF
from core.equations.synapses.CUBA import CUBA
from core.equations.neurons.fp8LIF import fp8LIF
from core.equations.synapses.fp8CUBA import fp8CUBA
from core.equations.neurons.int4LIF import int4LIF
from core.equations.synapses.int4CUBA import int4CUBA
from core.equations.neurons.int8LIF import int8LIF
from core.equations.synapses.int8CUBA import int8CUBA
from core.builder.groups_builder import create_synapses, create_neurons


def neuron_synapse_models(args):
    defaultclock.dt = args.timestep * ms

    """ ==================== Input ==================== """
    stim_dt = 1000
    stim_freq_tag = [8, 6, 4, 2, 1]
    stimulus = TimedArray(stim_freq_tag*Hz, dt=stim_dt*ms)
    poisson_pop = PoissonGroup(1000, 'stimulus(t)')

    """ ==================== Models ==================== """
    # Neurons
    # Each neuron is used to compute one "trial"
    n_neu = 10
    neuron_model = LIF()
    neuron_model.model += 'gtot1 : volt\n'
    layer1_fp64 = create_neurons(n_neu, neuron_model, name='std_neu1')

    neuron_model2 = fp8LIF()
    layer1_fp8 = create_neurons(n_neu, neuron_model2, name='fp8_neu1')

    neuron_model3 = int4LIF()
    layer1_int4 = create_neurons(n_neu, neuron_model3, name='int4_neu1')

    neuron_model4 = int8LIF()
    layer1_int8 = create_neurons(n_neu, neuron_model4, name='int8_neu1')

    # Synapses
    # Generating connections to be used for all models. Adjacent input
    # channels groups project only to a neuron "trial"
    sources = [x for x in range(poisson_pop.N)]
    targets = np.repeat(range(n_neu), poisson_pop.N/n_neu)

    synapse_model_test = CUBA()
    synapse_model_test.modify_model('connection', sources, key='i')
    synapse_model_test.modify_model('connection', targets, key='j')
    poisson_input_fp64 = create_synapses(poisson_pop, layer1_fp64,
                                         synapse_model_test,
                                         name='poisson_input_fp64')
    poisson_input_fp64.weight = 23.5*mV

    synapse_model_test2 = fp8CUBA()
    synapse_model_test2.modify_model('connection', sources, key='i')
    synapse_model_test2.modify_model('connection', targets, key='j')
    poisson_input_fp8 = create_synapses(poisson_pop, layer1_fp8,
                                        synapse_model_test2,
                                        name='poisson_input_fp8')
    poisson_input_fp8.weight = 97

    synapse_model_test3 = int4CUBA()
    synapse_model_test3.modify_model('connection', sources, key='i')
    synapse_model_test3.modify_model('connection', targets, key='j')
    poisson_input_int4 = create_synapses(poisson_pop, layer1_int4,
                                         synapse_model_test3,
                                         name='poisson_input_int4')
    poisson_input_int4.weight = 1

    synapse_model_test4 = int8CUBA()
    synapse_model_test4.modify_model('connection', sources, key='i')
    synapse_model_test4.modify_model('connection', targets, key='j')
    poisson_input_int8 = create_synapses(poisson_pop, layer1_int8,
                                         synapse_model_test4,
                                         name='poisson_input_int8')
    poisson_input_int8.weight = 16

    """ ==================== Monitors ==================== """
    spikemon_input = SpikeMonitor(poisson_pop)
    spikemon_layer1_fp64 = SpikeMonitor(layer1_fp64)
    statemon_input_synapse = StateMonitor(
        layer1_fp64, variables='gtot', record=range(n_neu))
    statemon_test_neurons1 = StateMonitor(layer1_fp64,
                                          variables=['Vm'],
                                          record=range(n_neu))

    spikemon_layer1_fp8 = SpikeMonitor(layer1_fp8)
    statemon_layer1_fp8 = StateMonitor(layer1_fp8, variables=[
        'gtot', 'Vm'], record=range(n_neu))

    spikemon_layer1_int8 = SpikeMonitor(layer1_int8)
    statemon_layer1_int8 = StateMonitor(layer1_int8, variables=[
        'gtot', 'Vm'], record=range(n_neu))

    spikemon_layer1_int4 = SpikeMonitor(layer1_int4)
    statemon_layer1_int4 = StateMonitor(layer1_int4, variables=[
        'gtot', 'Vm'], record=range(n_neu))

    """ ==================== running ==================== """
    duration = len(stim_freq_tag) * stim_dt
    run(duration * ms, namespace={'stimulus': stimulus}, profile=True)
    if args.backend == 'cpp_standalone':
        device.build(args.code_path)

    """ ==================== saving results ==================== """
    stim_rate_intervals = [stim_dt*i for i in range(len(stim_freq_tag) + 1)]
    metadata = {'stim_time_tag': stim_rate_intervals,
                'stim_freq_tag': stim_freq_tag}
    with open(args.save_path + 'metadata.json', 'w') as f:
        json.dump(metadata, f)

    rate_layer1_int4 = neurons_rate(spikemon_layer1_int4, duration)
    rate_layer1_int8 = neurons_rate(spikemon_layer1_int8, duration)
    rate_layer1_fp8 = neurons_rate(spikemon_layer1_fp8, duration)
    rate_layer1_fp64 = neurons_rate(spikemon_layer1_fp64, duration)
    temp_rate, temp_time, temp_res = [], [], []
    for dat in [(rate_layer1_int4, 'int4'), (rate_layer1_int8, 'int8'),
                (rate_layer1_fp8, 'fp8'), (rate_layer1_fp64, 'fp64')]:
        temp_rate.extend(dat[0][:, 0].magnitude.flatten())
        temp_time.extend(dat[0].times.magnitude)
        temp_res.extend([dat[1] for _ in range(len(dat[0].times))])
    rates = pd.DataFrame({'rate_Hz': temp_rate, 'times_ms': temp_time,
                          'resolution': temp_res})
    feather.write_dataframe(rates, args.save_path + 'rates.feather')

    # Non-refractory states are normalized to between 0-1 for comparison
    fp8_gtot_norm = [minifloat2decimal(x) / max(minifloat2decimal(x))
                        for x in statemon_layer1_fp8.gtot]
    fp8_vm_norm = [minifloat2decimal(x) / max(minifloat2decimal(x))
                   for x in statemon_layer1_fp8.Vm]

    fp64_gtot_norm = [x / max(x) for x in statemon_input_synapse.gtot]
    fp64_vm_norm = [x / max(x) for x in statemon_test_neurons1.Vm]

    int8_gtot_norm = [x / max(x) for x in statemon_layer1_int8.gtot]
    int8_vm_norm = [np.clip((x - layer1_int8.namespace['Vrest'])
                             / (max(x) - layer1_int8.namespace['Vrest']),
                            0, 1)
                        for x in statemon_layer1_int8.Vm]

    int4_gtot_norm = [x / max(x) for x in statemon_layer1_int4.gtot]
    int4_vm_norm = [np.clip((x - layer1_int4.namespace['Vrest'])
                             / (max(x) - layer1_int4.namespace['Vrest']),
                            0, 1)
                        for x in statemon_layer1_int4.Vm]

    temp_type, temp_res, temp_vals, temp_time, temp_id = [], [], [], [], []
    data = [(int4_gtot_norm, statemon_layer1_int4.t, 'int4', 'gtot'),
            (int4_vm_norm, statemon_layer1_int4.t, 'int4', 'vm'),
            (int8_gtot_norm, statemon_layer1_int8.t, 'int8', 'gtot'),
            (int8_vm_norm, statemon_layer1_int8.t, 'int8', 'vm'),
            (fp8_gtot_norm, statemon_layer1_fp8.t, 'fp8', 'gtot'),
            (fp8_vm_norm, statemon_layer1_fp8.t, 'fp8', 'vm'),
            (fp64_gtot_norm, statemon_input_synapse.t, 'fp64', 'gtot'),
            (fp64_vm_norm, statemon_input_synapse.t, 'fp64', 'vm')]
    for dat in data:
        for idx in range(np.shape(dat[0])[0]):
            temp_type.extend([dat[3] for _ in range(np.shape(dat[1])[0])])
            temp_res.extend([dat[2] for _ in range(np.shape(dat[1])[0])])
            temp_vals.extend(dat[0][idx])
            temp_time.extend(dat[1]/defaultclock.dt)
            temp_id.extend([idx for _ in range(len(dat[0][idx]))])
    traces = pd.DataFrame({'type': temp_type, 'resolution': temp_res,
                           'values': temp_vals, 'time_ms': temp_time,
                           'id': temp_id})
    feather.write_dataframe(traces, args.save_path + 'traces.feather')

    temp_input_rate, temp_pair, temp_coef, temp_trial = [], [], [], []
    for dat in zip(['int4', 'int8', 'fp8'],
                   [int4_gtot_norm, int8_gtot_norm, fp8_gtot_norm]):
        for i in range(len(stim_rate_intervals) - 1):
            for trial in range(np.shape(dat[1])[0]):
                temp_pair.append(f'{dat[0]}')
                temp_input_rate.append(str(stim_freq_tag[i]))
                temp_trial.append(trial)
                aux_val1 = fp64_gtot_norm[trial][stim_rate_intervals[i]:stim_rate_intervals[i+1]]
                aux_val2 = dat[1][trial][stim_rate_intervals[i]:stim_rate_intervals[i+1]]
                temp_coef.append(pearsonr(aux_val1, aux_val2)[0])
    corr = pd.DataFrame({'input_rate': temp_input_rate, 'pair': temp_pair,
                         'coef': temp_coef, 'trial': temp_trial})
    feather.write_dataframe(corr, args.save_path + 'corr.feather')

    int4_binned_spikes = monitor2binnedneo(spikemon_layer1_int4,
                                           time_interval=duration, bin_size=1)
    int8_binned_spikes = monitor2binnedneo(spikemon_layer1_int8,
                                           time_interval=duration, bin_size=1)
    fp8_binned_spikes = monitor2binnedneo(spikemon_layer1_fp8,
                                          time_interval=duration, bin_size=1)
    fp64_binned_spikes = monitor2binnedneo(spikemon_layer1_fp64,
                                           time_interval=duration, bin_size=1)
    temp_cch, temp_lags, temp_res, temp_input_rate = [], [], [], []
    for dat in zip(['int4', 'int8', 'fp8'],
                   [int4_binned_spikes, int8_binned_spikes, fp8_binned_spikes]):
        for i in range(len(stim_rate_intervals) - 1):
            avg_cch = []
            for trial in range(np.shape(dat[1])[0]):
                cch, lags = cross_correlation_histogram(
                                fp64_binned_spikes[trial].time_slice(
                                    stim_rate_intervals[i]*q.ms,
                                    stim_rate_intervals[i+1]*q.ms),
                                dat[1][trial].time_slice(
                                    stim_rate_intervals[i]*q.ms,
                                    stim_rate_intervals[i+1]*q.ms),
                                window=[-10, 10],
                                cross_correlation_coefficient=True)
                avg_cch.append(cch.magnitude.flatten())
            avg_cch = np.mean(avg_cch, axis=0)
            temp_cch.extend(avg_cch)
            temp_lags.extend(lags)
            temp_res.extend([dat[0] for _ in range(len(avg_cch))])
            temp_input_rate.extend([stim_freq_tag[i] for _ in range(len(avg_cch))])
    cch = pd.DataFrame({'cch': temp_cch, 'lags': temp_lags,
                        'resolution': temp_res, 'input_rate': temp_input_rate})
    feather.write_dataframe(cch, args.save_path + 'cch.feather')

    if not args.quiet:
        fig, axs = plt.subplots(3, 1, sharex=True)

        axs[0].title.set_text('input spikes')
        brian_plot(spikemon_input, axes=axs[0])

        axs[1].title.set_text('PSCs on 1st layer')
        plot_state(statemon_layer1_fp8.t,
                   fp8_gtot_norm[0],
                   var_name='gtot', label='fp8', axes=axs[1])
        plot_state(statemon_input_synapse.t,
                   fp64_gtot_norm[0],
                   var_name='gtot', label='fp64', axes=axs[1])
        plot_state(statemon_layer1_int8.t,
                   int8_gtot_norm[0],
                   var_name='gtot', label='int8', axes=axs[1])
        plot_state(statemon_layer1_int4.t,
                   int4_gtot_norm[0],
                   var_name='gtot', label='int4', axes=axs[1])

        axs[2].title.set_text('Vm on 1st layer')
        plot_state(statemon_layer1_fp8.t,
                   fp8_vm_norm[0],
                   var_name='Vm', label='fp8', axes=axs[2])
        plot_state(statemon_test_neurons1.t,
                   fp64_vm_norm[0],
                   var_name='Vm', label='fp64', axes=axs[2])
        plot_state(statemon_layer1_int8.t,
                   int8_vm_norm[0],
                   var_name='Vm', label='int8', axes=axs[2])
        plot_state(statemon_layer1_int4.t,
                   int4_vm_norm[0],
                   var_name='Vm', label='int4', axes=axs[2])
        plt.legend()
        plt.savefig(f'{args.save_path}/fig1')

        plt.figure()
        plot_rate(rate_layer1_fp64.times, rate_layer1_fp64[:, 0].magnitude.flatten(), label='fp64')
        plot_rate(rate_layer1_fp8.times, rate_layer1_fp8[:, 0].magnitude.flatten(), label='fp8')
        plot_rate(rate_layer1_int8.times, rate_layer1_int8[:, 0].magnitude.flatten(), label='int8')
        plot_rate(rate_layer1_int4.times, rate_layer1_int4[:, 0].magnitude.flatten(), label='int4')
        plt.legend()
        plt.savefig(f'{args.save_path}/fig2')
