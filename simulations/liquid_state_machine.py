from brian2 import SpikeMonitor, StateMonitor, SpikeGeneratorGroup
from brian2 import run, ms, seed
from brian2 import get_device, device

from core.utils.misc import decimal2minifloat, minifloat2decimal

from core.equations.neurons.fp8LIF import fp8LIF
from core.equations.synapses.fp8CUBA import fp8CUBA
from core.builder.groups_builder import create_synapses, create_neurons
from core.utils.testbench import create_item, create_sequence, create_testbench

import numpy as np
import matplotlib.pyplot as plt
import json

import neo
import quantities as q
from elephant import statistics, kernels
from elephant.statistics import isi, cv

from viziphant.statistics import plot_instantaneous_rates_colormesh
from brian2tools import brian_plot, plot_state


def liquid_state_machine(defaultclock, trial_no, path, quiet):
    item_rate = 128
    repetitions = 30
    inter_spk_interval = np.ceil(1/item_rate*1000).astype(int)
    item_spikes = 3
    A = create_item([0], inter_spk_interval, item_spikes)
    B = create_item([1], inter_spk_interval, item_spikes)
    C = create_item([2], inter_spk_interval, item_spikes)
    D = create_item([3], inter_spk_interval, item_spikes)
    E = create_item([4], inter_spk_interval, item_spikes)
    F = create_item([5], inter_spk_interval, item_spikes)
    G = create_item([6], inter_spk_interval, item_spikes)
    H = create_item([7], inter_spk_interval, item_spikes)

    seq1 = [A, B, C, D, E, F, G, H]
    seq2 = [H, G, F, E, D, C, B, A]
    seq1 = create_sequence(seq1, 0)
    seq2 = create_sequence(seq2, 0)

    channels_per_item = 1
    num_items = 8
    num_seq = 2
    print(f'Simulation with {num_seq} sequences, each having {num_items} '
          f'items represented by {channels_per_item} input channels')

    input_indices, input_times, events = create_testbench([seq1, seq2],
                                                          [.5, .5],
                                                          40,
                                                          repetitions)
    input_indices = np.array(input_indices)
    input_times = np.array(input_times) * ms
    sequence_duration = max(seq1['times']) * ms
    num_channels = int(max(input_indices) + 1)
    sim_dur = np.max(input_times)
    test_dur = 1000*ms
    input_spikes = SpikeGeneratorGroup(num_channels,
                                       input_indices,
                                       input_times)

    # TODO sizes from 128, 256, 512, 1024, 2048, 4096. Original was 4084
    Nt = 4096
    Ne, Ni = np.rint(Nt*.85).astype(int), np.rint(Nt*.15).astype(int)

    neu_model = fp8LIF()
    cells = create_neurons(Ne+Ni, neu_model)
    exc_cells = cells[:Ne]
    inh_cells = cells[Ne:]

    e_syn_model = fp8CUBA()
    # TODO weight was 10(dec) and p .25
    e_syn_model.connection['p'] = .12
    e_syn_model.modify_model('parameters', decimal2minifloat(56), key='weight')
    thl_conns = create_synapses(input_spikes, cells, e_syn_model)

    e_syn_model = fp8CUBA()
    e_syn_model.connection['p'] = .1
    intra_exc = create_synapses(exc_cells, cells, e_syn_model)

    i_syn_model = fp8CUBA()
    i_syn_model.connection['p'] = .1
    i_syn_model.namespace['w_factor'] = decimal2minifloat(-1)
    i_syn_model.parameters['weight'] = 110# TODO decimal2minifloat(48)
    intra_inh = create_synapses(inh_cells, cells, i_syn_model)

    e_neu_model = fp8LIF()
    e_neu_model.modify_model('parameters', 55, key='alpha_refrac')
    readout = create_neurons(num_seq, e_neu_model, name='readout')

    labels_indices = []
    labels_times = []
    for ev in events:
        labels_indices.append(ev[0])
        labels_times.append(ev[2])
    labels = SpikeGeneratorGroup(num_seq, labels_indices, labels_times)

    e_syn_model = fp8CUBA()
    e_syn_model.model += 'delta_t : second\ndelay_proxy : second\n'
    # TODO delays with 129 are to be pruned
    e_syn_model.on_post += f'delta_t = clip(t - lastspike_pre, 0*ms, {sequence_duration/ms + 1}*ms)\n'
    e_syn_model.on_post += f'delay_proxy = delay_proxy - .1*(delay_proxy - delta_t)\n'
    e_syn_model.parameters = {**e_syn_model.parameters, 'delay_proxy': '0*ms'}
    # TODO below only for large net
    e_syn_model.modify_model('parameters', 1, key='weight')
    exc_readout = create_synapses(exc_cells, readout, e_syn_model,
                                   name='exc_readout')

    e_syn_model = fp8CUBA()
    e_syn_model.modify_model('connection', 'i', key='j')
    e_syn_model.modify_model('parameters', 115, key='weight')
    label_readout = create_synapses(labels, readout, e_syn_model,
                                     name='label_readout')

    selected_exc_cells = np.random.choice(Ne, 4, replace=False)
    selected_inh_cells = np.random.choice(Ni, 4, replace=False)

    Metadata = {'selected_exc_cells': selected_exc_cells.tolist(),
                'selected_inh_cells': selected_inh_cells.tolist(),
                'dt': str(defaultclock.dt),
                'trial_no': trial_no,
                'duration': str(sim_dur*ms),
                'inh_weight': i_syn_model.parameters['weight']}
    with open(path+'metadata.json', 'w') as f:
        json.dump(Metadata, f)

    spkmon_e = SpikeMonitor(exc_cells)
    spkmon_i = SpikeMonitor(inh_cells)
    spkmon_ro = SpikeMonitor(readout)
    sttmon_e = StateMonitor(exc_cells, variables='Vm',
                            record=selected_exc_cells)
    sttmon_i = StateMonitor(inh_cells, variables='Vm',
                            record=selected_inh_cells)
    sttmon_ro = StateMonitor(readout, variables='Vm',
                            record=[0, 1])

    kernel = kernels.GaussianKernel(sigma=30*q.ms)
    run(sim_dur-test_dur)

    exc_readout.delay = 'delay_proxy'
    # TODO e_neu_model.modify_model('namespace', decimal2minifloat(30), key='Vthr')
    # TODO small weights e.g. 10 get stuck, high explodes e.g. 56. 1 kindda works. 60 for small net
    #e_syn_model.modify_model('parameters', 60, key='weight')
    label_readout.namespace['w_factor'] = 0
    run(test_dur)
    device.build()

    fig,  (ax0, ax1, ax2) = plt.subplots(3, 1, sharex=True)
    plot_state(sttmon_ro.t, minifloat2decimal(sttmon_ro.Vm[0]), axes=ax0)
    plot_state(sttmon_ro.t, minifloat2decimal(sttmon_ro.Vm[1]), axes=ax0)
    brian_plot(spkmon_e, axes=ax1)
    ax2.plot(input_times/ms, input_indices, '.')
    plt.show()

    temp_trains = spkmon_e.spike_trains()
    spk_trains = [neo.SpikeTrain(temp_trains[x]/ms, t_stop=sim_dur/ms, units='ms')
                  for x in temp_trains]
    pop_rates = statistics.instantaneous_rate(spk_trains,
                                              sampling_period=1*q.ms,
                                              kernel=kernel)
    pop_avg_rates = np.mean(pop_rates, axis=1)

    np.savez(f'{path}/exc_raster.npz',
             times=spkmon_e.t/ms,
             indices=spkmon_e.i)
    np.savez(f'{path}/inh_raster.npz',
             times=spkmon_i.t/ms,
             indices=spkmon_i.i)
    np.savez(f'{path}/rates.npz',
             times=np.array(pop_rates.times),
             rates=np.array(pop_avg_rates))

    if not quiet:
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax2.plot(pop_rates.times, pop_avg_rates, color='red')
        # TODO brian_plot(spkmon_e, marker=',', color='black', axes=ax1)
        brian_plot(spkmon_e, color='black', axes=ax1)
        ax1.set_xlabel(f'time ({pop_rates.times.dimensionality.latex})')
        ax1.set_ylabel('neuron number')
        ax2.set_ylabel(f'rate ({pop_rates.dimensionality})')

        plot_instantaneous_rates_colormesh(pop_rates)
        plt.title('Neuron rates on last trial')

        isi_neu = [isi(spks) for spks in spk_trains]
        fig, ax3 = plt.subplots()
        flatten_isi = []
        for vals in isi_neu:
            flatten_isi = np.append(flatten_isi, vals)
        ax3.hist(flatten_isi, bins=np.linspace(-3, 100, 10))
        ax3.set_title('ISI distribution')
        ax3.set_xlabel('ISI')
        ax3.set_ylabel('count')

        plt.figure()
        cv_neu = [cv(x) for x in isi_neu]
        plt.hist(cv_neu)
        plt.title('Coefficient of variation')
        plt.ylabel('count')
        plt.xlabel('CV')

        plt.show()
