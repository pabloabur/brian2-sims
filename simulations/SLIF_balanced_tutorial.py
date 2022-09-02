from brian2 import PoissonGroup, SpikeMonitor, StateMonitor
from brian2 import defaultclock, prefs, Network, collect
from brian2 import second, Hz, ms, ohm, mA

from orca_column import orcaColumn
from teili.tools.misc import minifloat2decimal, decimal2minifloat

from utils.SLIF_utils import neuron_rate, get_metrics
from parameters.orca_params import ConnectionDescriptor, PopulationDescriptor

from equations.neurons.fp8LIF import fp8LIF
from equations.synapses.fp8CUBA import fp8CUBA
from builder.groups_builder import create_synapses, create_neurons

import sys
import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import pickle
import json
import argparse

import neo
import quantities as q
from elephant import statistics, kernels
from elephant.statistics import isi, cv

from viziphant.statistics import plot_instantaneous_rates_colormesh
from brian2tools import brian_plot

def change_intra_conn(desc):
    for conn in desc.plasticities.keys():
        desc.plasticities[conn] = 'static'
    for conn in desc.probabilities.keys():
        desc.probabilities[conn] = 0.1
    desc.filter_params()
    for conn in desc.sample.keys():
        desc.sample[conn] = []
    max_weight = 2**(desc.constants['n_bits'] - 1) - 1
    for conn in desc.base_vals:
        if conn[:3]=='pyr':
            desc.base_vals[conn]['weight'] = np.floor(.2*max_weight)
        elif conn[:3]=='pv_' or conn[:3]=='vip' or conn[:3]=='sst':
            # TODO
            if wi>1:
                raise TypeError
            desc.base_vals[conn]['weight'] = np.floor(-wi*max_weight)

def change_pop(desc):
    for pop in desc.group_plast.keys():
        desc.group_plast[pop] = 'static'
    desc.group_prop['n_exc'] = Ne
    desc.group_prop['inh_ratio']['pv_cells'] = 1.0
    desc.group_prop['inh_ratio']['sst_cells'] = 0.0
    desc.group_prop['inh_ratio']['vip_cells'] = 0.0
    desc.filter_params()
    desc.groups['vip_cells']['num_inputs'] = 3
    for pop in desc.sample.keys():
        desc.sample[pop] = []

def change_input_conn(desc):
    for conn in desc.plasticities.keys():
        desc.plasticities[conn] = 'static'
    for conn in desc.probabilities.keys():
        desc.probabilities[conn] = 0.25
    desc.filter_params()
    for conn in desc.sample.keys():
        desc.sample[conn] = []
    max_weight = 2**(desc.constants['n_bits'] - 1) - 1
    for conn in desc.base_vals.keys():
        desc.base_vals[conn]['weight'] = np.floor(.42*max_weight)

parser = argparse.ArgumentParser(description='Reproduces a balanced network')
parser.add_argument('wi', type=float, help='strength of inhibitory weight')
parser.add_argument('--path', default=None, help='directory to save')
parser.add_argument('--quiet', action='store_true',
                    help='whether to create plots/info or not')
args = parser.parse_args()
wi = args.wi
path = args.path
quiet = args.quiet

defaultclock.dt = 1*ms
prefs.codegen.target = "numpy"
sim_dur = 1000

poisson_spikes = PoissonGroup(285, rates=6*Hz)

Ne, Ni = 3471, 613

#layer = ['L4']
#column = orcaColumn(layer)
#conn_modifier = {'L4': change_intra_conn}
#pop_modifier = {'L4': change_pop}
#column.create_layers(pop_modifier, conn_modifier)

neu_model = fp8LIF()
cells = create_neurons(Ne+Ni, neu_model)
exc_cells = cells[:Ne]
inh_cells = cells[Ne:]

# Feedforward weights elicited only a couple of spikes on 
# excitatory neurons
#conn_modifier = {'L4': change_input_conn}
#column.connect_inputs(poisson_spikes, 'ff', conn_modifier)
#column.col_groups['L4'].groups['pyr_pyr'].delay = 0*ms

e_syn_model = fp8CUBA()
e_syn_model.connection['p'] = .25
thl_conns = create_synapses(poisson_spikes, cells, e_syn_model)
e_syn_model = fp8CUBA()
e_syn_model.connection['p'] = .1
intra_exc = create_synapses(exc_cells, cells, e_syn_model)
i_syn_model = fp8CUBA()
i_syn_model.connection['p'] = .1
i_syn_model.namespace['w_factor'] = decimal2minifloat(-1)
i_syn_model.parameters['weight'] = decimal2minifloat(wi)
intra_inh = create_synapses(inh_cells, cells, i_syn_model)

#spkmon_e = SpikeMonitor(column.col_groups['L4'].groups['pyr_cells'])
#spkmon_i = SpikeMonitor(column.col_groups['L4'].groups['pv_cells'])
#sttmon_e = StateMonitor(column.col_groups['L4'].groups['pyr_cells'],
#                        variables='Vm', record=np.random.choice(Ne, 4, replace=False))
#sttmon_i = StateMonitor(column.col_groups['L4'].groups['pv_cells'],
#                        variables='Vm', record=np.random.choice(Ni, 4, replace=False))

trials = 10
selected_exc_cells = np.random.choice(Ne, 4, replace=False)
selected_inh_cells = np.random.choice(Ni, 4, replace=False)

if not path:
    date_time = datetime.now()
    path = f"""{date_time.strftime('%Y.%m.%d')}_{date_time.hour}.{date_time.minute}/"""
    os.makedirs(path)

Metadata = {'selected_exc_cells': selected_exc_cells.tolist(),
            'selected_inh_cells': selected_inh_cells.tolist(),
            'dt': str(defaultclock.dt),
            'trials': trials,
            'duration': str(sim_dur*ms),
            'inh_weight': i_syn_model.parameters['weight']}
with open(f'{path}/metadata.json', 'w') as f:
    json.dump(Metadata, f)

spkmon_e = SpikeMonitor(exc_cells)
spkmon_i = SpikeMonitor(inh_cells)
sttmon_e = StateMonitor(exc_cells, variables='Vm',
                        record=selected_exc_cells)
sttmon_i = StateMonitor(inh_cells, variables='Vm',
                        record=selected_inh_cells)

trial_avg_rate = []
kernel = kernels.GaussianKernel(sigma=30*q.ms)
net = Network(collect())
#net.add([x for x in column.col_groups.values()])
#net.add([x.input_groups for x in column.col_groups.values()])
net.store()
for trial in range(trials):
    print('########################')
    print(f'Starting trial {trial+1}')
    net.restore()
    net.run(sim_dur*ms)

    temp_trains = spkmon_e.spike_trains()
    spk_trains = [neo.SpikeTrain(temp_trains[x]/ms, t_stop=sim_dur, units='ms')
                  for x in temp_trains]
    pop_rates = statistics.instantaneous_rate(spk_trains,
                                              sampling_period=1*q.ms,
                                              kernel=kernel)
    pop_avg_rates = np.mean(pop_rates, axis=1)
    trial_avg_rate.append(np.mean(pop_avg_rates))

np.savez(f'{path}/exc_raster.npz',
         times=spkmon_e.t/ms,
         indices=spkmon_e.i)
np.savez(f'{path}/inh_raster.npz',
         times=spkmon_i.t/ms,
         indices=spkmon_i.i)
np.save(f'{path}/avg_rate.npy', trial_avg_rate)

if not quiet:
    # Only last trials considered below
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax2.plot(pop_rates.times, pop_avg_rates)
    brian_plot(spkmon_e, marker=',', axes=ax1)
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
