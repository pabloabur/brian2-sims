"""
This code implements a sequence learning using and Excitatory-Inhibitory
network with STDP.
"""
import numpy as np
import matplotlib.pyplot as plt

from brian2 import ms, second, mA, Hz, prefs, SpikeMonitor, StateMonitor,\
        SpikeGeneratorGroup, defaultclock, ExplicitStateUpdater,\
        PopulationRateMonitor, seed

from teili import TeiliNetwork
from teili.stimuli.testbench import SequenceTestbench
from teili.tools.misc import neuron_group_from_spikes

from SLIF_utils import neuron_rate, rate_correlations, ensemble_convergence,\
        permutation_from_rate, load_merge_multiple, recorded_bar_testbench,\
        run_batches

from orca_wta import orcaWTA
from monitor_params import monitor_params, selected_cells
from orca_params import ConnectionDescriptor, PopulationDescriptor

import sys
import pickle
import os
from datetime import datetime


# Initialize simulation preferences
seed(13)
rng = np.random.default_rng(12345)
prefs.codegen.target = "numpy"
defaultclock.dt = 1*ms

# Initialize input sequence
num_items = 4
item_duration = 60
item_superposition = 0
num_channels = 144
noise_prob = None#0.005
item_rate = 100
#repetitions = 700# 350
repetitions = 200

sequence = SequenceTestbench(num_channels, num_items, item_duration,
                             item_superposition, noise_prob, item_rate,
                             repetitions, surprise_item=True)
input_indices, input_times = sequence.stimuli()
training_duration = np.max(input_times)
sequence_duration = sequence.cycle_length * ms
testing_duration = 0*ms

# Adding alternative sequence at the end of simulation
alternative_sequences = 10
include_symbols = [[0, 1, 2, 3] for _ in range(alternative_sequences)]
test_duration = alternative_sequences * sequence_duration
symbols = sequence.items
for alt_seq in range(alternative_sequences):
    for incl_symb in include_symbols[alt_seq]:
        tmp_symb = [(x*ms + alt_seq*sequence_duration + training_duration)
                        for x in symbols[incl_symb]['t']]
        input_times = np.append(input_times, tmp_symb)
        input_indices = np.append(input_indices, symbols[incl_symb]['i'])
# Get back unit that was remove by append operation
input_times = input_times*second

# Adding noise at the end of simulation
#alternative_sequences = 5
#test_duration = alternative_sequences*sequence_duration
#noise_prob = 0.01
#noise_spikes = np.random.rand(num_channels, int(test_duration/ms))
#noise_indices = np.where(noise_spikes < noise_prob)[0]
#noise_times = np.where(noise_spikes < noise_prob)[1]
#input_indices = np.concatenate((input_indices, noise_indices))
#input_times = np.concatenate((input_times, noise_times+training_duration/ms))
# TODO sorting may be needed, as well as adjusting units above

training_duration = np.max(input_times) - testing_duration
sim_duration = input_times[-1]
# Convert input into neuron group (necessary for STDP compatibility)
relay_cells = neuron_group_from_spikes(num_channels,
                                       defaultclock.dt,
                                       sim_duration,
                                       spike_indices=input_indices,
                                       spike_times=input_times)

Net = TeiliNetwork()
model_path = '/Users/Pablo/git/teili/'
layer='L4'
conn_desc = ConnectionDescriptor(layer, model_path)
pop_desc = PopulationDescriptor(layer, model_path)
# TODO not working with altadp
conn_desc.intra_plast['sst_pv'] = 'static'
pop_desc.e_ratio = .00224
pop_desc.group_prop['inh_ratio']['pv_cells'] = .75
pop_desc.group_prop['inh_ratio']['sst_cells'] = .125
pop_desc.group_prop['inh_ratio']['vip_cells'] = .125
# TODO just testing with more I
for pop in pop_desc.group_vals:
    pop_desc.group_vals[pop]['I_min'] = -256*mA
    pop_desc.group_vals[pop]['I_max'] = 256*mA
conn_desc.intra_plast['pyr_pyr'] = 'redsymstdp'
conn_desc.input_plast['ff_pyr'] = 'redsymstdp'
conn_desc.filter_params()
pop_desc.filter_params()
num_exc = pop_desc._groups['pyr_cells']['num_neu']
orca = orcaWTA(layer=layer,
               conn_params=conn_desc,
               pop_params=pop_desc,
               monitor=True)
re_init_dt = None

orca.add_input(relay_cells, 'ff', ['pyr_cells', 'pv_cells'])
# TODO remove this test
from brian2 import mA
#orca2._groups['ff_pyr'].gain_syn = 8*mA
#orca2._groups['ff_pv'].gain_syn = 8*mA
#orca2._groups['ff_sst'].gain_syn = 8*mA
#orca2._groups['sst_pyr'].gain_syn = 4*mA
#orca2._groups['pv_pyr'].gain_syn = 4*mA

# Prepare for saving data
date_time = datetime.now()
path = f"""{date_time.strftime('%Y.%m.%d')}_{date_time.hour}.{date_time.minute}/"""
os.mkdir(path)
Metadata = {'time_step': defaultclock.dt,
            'num_exc': num_exc,
            'num_pv': orca._groups['pv_cells'].N,
            'num_channels': num_channels,
            'full_rotation': sequence_duration,
            'repetitions': repetitions,
            'selected_cells': selected_cells,
            're_init_dt': re_init_dt
        }

with open(path+'metadata', 'wb') as f:
    pickle.dump(Metadata, f)

##################
# Setting up monitors
orca.create_monitors(monitor_params)

# Temporary monitors
statemon_net_current = StateMonitor(orca._groups['pyr_cells'],
    variables=['Iin', 'Iin0', 'Iin1', 'Iin2', 'Iin3', 'I', 'Vm'], record=True,
    name='statemon_net_current')
statemon_net_current2 = StateMonitor(orca._groups['pv_cells'],
    variables=['Iin'], record=True,
    name='statemon_net_current2')
statemon_net_current3 = StateMonitor(orca._groups['sst_cells'],
    variables=['Iin'], record=True,
    name='statemon_net_current3')
spikemon_input = SpikeMonitor(relay_cells, name='input_spk')

# Training
Net.add(orca, relay_cells)
Net.add(statemon_net_current, statemon_net_current2, statemon_net_current3, spikemon_input)

training_blocks = 10
run_batches(Net, orca, training_blocks, training_duration,
            defaultclock.dt, path, monitor_params)

# Testing network
if testing_duration:
    block += 1
    orca._groups['ff_pyr'].stdp_thres = 0
    orca._groups['pyr_pyr'].stdp_thres = 0
    orca._groups['sst_pyr'].inh_learning_rate = 0
    orca._groups['pv_pyr'].inh_learning_rate = 0

    # deactivate bottom-up
    orca._groups['ff_pyr'].weight = 0
    orca._groups['ff_pv'].weight = 0
    orca._groups['ff_sst'].weight = 0
    Net.run(testing_duration, report='stdout', report_period=100*ms)
    orca.save_data(monitor_params, path, block)

# Recover data pickled from monitor
spikemon_exc_neurons = load_merge_multiple(path, 'pickled*')

last_sequence_t = training_duration - int(sequence_duration/num_items/defaultclock.dt)*defaultclock.dt*3
neu_rates = neuron_rate(spikemon_exc_neurons, kernel_len=10*ms,
    kernel_var=1*ms, simulation_dt=defaultclock.dt,
    interval=[last_sequence_t, training_duration], smooth=True,#)
    trials=3)

############
# Saving results
# Calculating permutation indices from firing rates
permutation_ids = permutation_from_rate(neu_rates)

# Save data
np.savez(path+f'input_raster.npz',
         input_t=np.array(spikemon_input.t/ms),
         input_i=np.array(spikemon_input.i)
        )

np.savez(path+f'permutation.npz',
         ids = permutation_ids
        )
  
Net.store(filename=f'{path}network')
