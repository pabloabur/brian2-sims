import numpy as np

import matplotlib.pylab as plt

from brian2 import ms, mA, second, prefs, SpikeMonitor,\
        defaultclock, seed, store, restore

from teili import TeiliNetwork
from teili.tools.misc import neuron_group_from_spikes

from SLIF_utils import deterministic_sequence

from orca_column import orcaColumn
from monitor_params import monitor_params, selected_cells

import pickle
import os
from datetime import datetime


def change_params_conn1(desc):
    # Changes some intralaminar connections
    desc.plasticities['sst_pv'] = 'static'
    desc.plasticities['pyr_pyr'] = 'redsymstdp'
    desc.probabilities['pyr_pyr'] = .3
    desc.filter_params()
    desc.base_vals['pyr_pyr']['w_max'] = 15
    desc.base_vals['pyr_pyr']['stdp_thres'] = 15

def change_params_conn2(desc):
    # Changes input parameters
    desc.plasticities['ff_pyr'] = 'redsymstdp'
    desc.filter_params()
    desc.base_vals['ff_pyr']['w_max'] = 15
    desc.base_vals['ff_pyr']['stdp_thres'] = 7

def change_params_conn3(desc):
    # Changes input parameters
    desc.plasticities['ff_pyr'] = 'redsymstdp'
    desc.probabilities['ff_pyr'] = 0.7
    desc.probabilities['ff_pv'] = 1
    desc.probabilities['ff_sst'] = 1
    desc.filter_params()
    desc.base_vals['ff_pyr']['w_max'] = 15
    desc.base_vals['ff_pyr']['stdp_thres'] = 7

def change_params_conn4(desc):
    # Changes interlaminar parameters
    desc.plasticities['pyr_pyr'] = 'redsymstdp'
    desc.probabilities['pyr_pyr'] = .3
    desc.filter_params()
    desc.base_vals['pyr_pyr']['w_max'] = 15
    desc.base_vals['pyr_pyr']['stdp_thres'] = 7

def change_params_pop1(desc):
    # Changes proportions of the EI population
    desc.e_ratio = .00224
    desc.filter_params()
    for pop in desc.base_vals:
        desc.base_vals[pop]['I_min'] = -256*mA
        desc.base_vals[pop]['I_max'] = 256*mA

def change_params_pop2(desc):
    # Changes proportions of the EI population
    desc.e_ratio = .01
    desc.group_prop['ei_ratio'] = 4
    desc.group_prop['inh_ratio']['pv_cells'] = .68
    desc.group_prop['inh_ratio']['sst_cells'] = .20
    desc.group_prop['inh_ratio']['vip_cells'] = .12
    desc.filter_params()
    for pop in desc.base_vals:
        desc.base_vals[pop]['I_min'] = -256*mA
        desc.base_vals[pop]['I_max'] = 256*mA

def testing_code(orca):
    orca.groups['pyr_pyr'].stdp_thres = 0
    orca.groups['pyr_cells'].tau_thr = 3600*second
    orca.groups['sst_pyr'].stdp_thres = 0
    orca.groups['pv_pyr'].stdp_thres = 0
    orca.groups['pv_cells'].tau_thr = 3600*second

# Initialize simulation preferences
seed(13)
rng = np.random.default_rng(12345)
prefs.codegen.target = "numpy"
defaultclock.dt = 1*ms

# Initialize input sequence
num_items = 9
item_duration = 20
item_superposition = 5
num_channels = 36
noise_prob = None
item_rate = 250
repetitions = 300

spike_indices = []
spike_times = []
for i in range(repetitions):
    sequence = deterministic_sequence(num_channels, num_items, item_duration,
                                 item_superposition, noise_prob, item_rate,
                                 1)
    tmp_i, tmp_t = sequence.stimuli()

    spike_indices.extend(tmp_i)
    tmp_t = [x/ms + i*(num_items*(item_duration - item_superposition)+item_superposition)
        for x in tmp_t]
    spike_times.extend(tmp_t)
input_indices = np.array(spike_indices)
input_times = np.array(spike_times) * ms
sequence_duration = sequence.cycle_length * ms
testing_duration = 1000*ms
training_duration = np.max(input_times)

# Convert input into neuron group (necessary for STDP compatibility)
relay_cells = neuron_group_from_spikes(num_channels,
                                       defaultclock.dt,
                                       training_duration,
                                       spike_indices=input_indices,
                                       spike_times=input_times)

Net = TeiliNetwork()
column = orcaColumn(['L4', 'L5'])
conn_modifier = {'L4': change_params_conn1, 'L5': change_params_conn1}
pop_modifier = {'L4': change_params_pop1, 'L5': change_params_pop2}
column.create_layers(pop_modifier, conn_modifier)
conn_modifier = {'L4_L5': change_params_conn4}
column.connect_layers(conn_modifier)
conn_modifier = {'L4': change_params_conn2, 'L5': change_params_conn3}
column.connect_inputs(relay_cells, 'ff', conn_modifier)

# Prepare for saving data
date_time = datetime.now()
path = f"""{date_time.strftime('%Y.%m.%d')}_{date_time.hour}.{date_time.minute}/"""
os.mkdir(path)
num_exc = column.col_groups['L4'].groups['pyr_cells'].N
Metadata = {'time_step': defaultclock.dt,
            'num_exc': num_exc,
            'num_pv': column.col_groups['L4'].groups['pv_cells'].N,
            'num_channels': num_channels,
            'full_rotation': sequence_duration,
            'repetitions': repetitions,
            'selected_cells': selected_cells,
            're_init_dt': None
        }

with open(path+'metadata', 'wb') as f:
    pickle.dump(Metadata, f)

##################
# Setting up monitors
monitor_params['statemon_static_conn_ff_pyr']['group'] = 'L4_ff_pyr'
monitor_params['statemon_conn_ff_pv']['group'] = 'L4_ff_pv'
monitor_params['statemon_static_conn_ff_pv']['group'] = 'L4_ff_pv'
monitor_params['statemon_conn_ff_pyr']['group'] = 'L4_ff_pyr'
column.col_groups['L4'].create_monitors(monitor_params)

# Temporary monitors
spikemon_input = SpikeMonitor(relay_cells, name='input_spk')
spkmon_l5 = SpikeMonitor(column.col_groups['L5']._groups['pyr_cells'],
                         name='l5_spk')
spkmon_l4 = SpikeMonitor(column.col_groups['L4']._groups['pyr_cells'],
                         name='l4_spk')

# Training
Net.add([x for x in column.col_groups.values()])
Net.add([x.input_groups for x in column.col_groups.values()])
Net.add(spikemon_input, spkmon_l4, spkmon_l5)

# store in a first simulation
#Net.store(filename='network')
# restore in a second simulation
Net.restore(filename='network')

Net.run(1000*ms, report='stdout', report_period=100*ms)
