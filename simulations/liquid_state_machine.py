from brian2 import PoissonGroup, SpikeMonitor, StateMonitor, SpikeGeneratorGroup
from brian2 import defaultclock, prefs, Network, collect, device, get_device,\
        set_device, run
from brian2 import second, Hz, ms, ohm, mA

from core.utils.misc import minifloat2decimal, decimal2minifloat

from core.utils.SLIF_utils import neuron_rate, get_metrics
from core.parameters.orca_params import ConnectionDescriptor, PopulationDescriptor

from core.equations.neurons.fp8LIF import fp8LIF
from core.equations.synapses.fp8CUBA import fp8CUBA
from core.builder.groups_builder import create_synapses, create_neurons
from core.utils.testbench import create_item, create_sequence, create_testbench

import sys
import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import json
import argparse

import neo
import quantities as q
from elephant import statistics, kernels
from elephant.statistics import isi, cv

from viziphant.statistics import plot_instantaneous_rates_colormesh
from brian2tools import brian_plot

parser = argparse.ArgumentParser(description='LSM with distinct resolutions')
parser.add_argument('--trial', type=int, default=0, help='trial number')
parser.add_argument('--path', default=None, help='directory to save')
parser.add_argument('--quiet', action='store_true',
                    help='whether to create plots/info or not')
args = parser.parse_args()
trial_no = args.trial
path = args.path
quiet = args.quiet

defaultclock.dt = 1*ms
#prefs.codegen.target = "numpy"
set_device('cpp_standalone', directory='output_balance')

item_rate = 128
repetitions = 200
isi = np.ceil(1/item_rate*1000).astype(int)
item_spikes = 3
A = create_item([0], isi, item_spikes)
B = create_item([1], isi, item_spikes)
C = create_item([2], isi, item_spikes)
D = create_item([3], isi, item_spikes)
E = create_item([4], isi, item_spikes)
F = create_item([5], isi, item_spikes)
G = create_item([6], isi, item_spikes)
H = create_item([7], isi, item_spikes)

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
input_spikes = SpikeGeneratorGroup(num_channels, input_indices, input_times)

# TODO sizes from 128, 256, 512, 1024, 2048, 4096
Ne, Ni = 3471, 613

neu_model = fp8LIF()
cells = create_neurons(Ne+Ni, neu_model)
exc_cells = cells[:Ne]
inh_cells = cells[Ne:]

e_syn_model = fp8CUBA()
e_syn_model.connection['p'] = .25
thl_conns = create_synapses(input_spikes, cells, e_syn_model)

e_syn_model = fp8CUBA()
e_syn_model.connection['p'] = .1
intra_exc = create_synapses(exc_cells, cells, e_syn_model)

i_syn_model = fp8CUBA()
i_syn_model.connection['p'] = .1
i_syn_model.namespace['w_factor'] = decimal2minifloat(-1)
i_syn_model.parameters['weight'] = 100
intra_inh = create_synapses(inh_cells, cells, i_syn_model)

selected_exc_cells = np.random.choice(Ne, 4, replace=False)
selected_inh_cells = np.random.choice(Ni, 4, replace=False)

if not path:
    date_time = datetime.now()
    path = f"""{date_time.strftime('%Y.%m.%d')}_{date_time.hour}.{date_time.minute}/"""
os.makedirs(path)

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
sttmon_e = StateMonitor(exc_cells, variables='Vm',
                        record=selected_exc_cells)
sttmon_i = StateMonitor(inh_cells, variables='Vm',
                        record=selected_inh_cells)

kernel = kernels.GaussianKernel(sigma=30*q.ms)
run(sim_dur)

temp_trains = spkmon_e.spike_trains()
spk_trains = [neo.SpikeTrain(temp_trains[x]/ms, t_stop=sim_dur, units='ms')
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
    brian_plot(spkmon_e, marker=',', color='black', axes=ax1)
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
