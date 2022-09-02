"""
This code implements a sequence learning using and Excitatory-Inhibitory
network with STDP.
"""
import numpy as np
import matplotlib.pyplot as plt

from brian2 import ms, mA, second, Hz, prefs, SpikeMonitor, StateMonitor,\
        SpikeGeneratorGroup, defaultclock, ExplicitStateUpdater,\
        PopulationRateMonitor, seed

from teili import TeiliNetwork
from teili.stimuli.testbench import OCTA_Testbench
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

# Initialize rotating bar
sequence_duration = 105*ms#357*ms#
#sequence_duration = 950*ms#1900*ms#
testing_duration = 0*ms
repetitions = 400
#repetitions = 13#8#10#
num_samples = 100
num_channels = num_samples
# Simulated bar
testbench_stim = OCTA_Testbench()
testbench_stim.rotating_bar(length=10, nrows=10,
                            direction='cw',
                            ts_offset=3, angle_step=10,#3,#
                            #noise_probability=0.2,
                            repetitions=repetitions,
                            debug=False)
input_indices = testbench_stim.indices
input_times = testbench_stim.times * ms
# Recorded bar
#input_times, input_indices = recorded_bar_testbench('../raw_7p5V_normallight_fullbar.aedat4_events.npz', num_samples, repetitions)

training_duration = np.max(input_times) - testing_duration
sim_duration = input_times[-1]
# Convert input into neuron group (necessary for STDP compatibility)
ff_cells = neuron_group_from_spikes(num_channels,
                                    defaultclock.dt,
                                    sim_duration,
                                    spike_indices=input_indices,
                                    spike_times=input_times)

# TODO values below were used before. New parameters need testing
# 'pyr_pyr': 0.5, # this has been standard. Commented is H&S
# 'pyr_pv': 0.15, # 0.45
# 'pyr_sst': 0.15, # 0.35
# 'pyr_vip': 0.10, # 0.10
# 'pv_pyr': 1.0, # 0.60
# 'pv_pv': 1.0, # 0.50
# 'sst_pv': 0.9, # 0.60
# 'sst_pyr': 1.0, # 0.55
# 'sst_vip': 0.9, # 0.45
# 'vip_sst': 0.65}, # 0.50
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

orca.add_input(ff_cells, 'ff', ['pyr_cells', 'pv_cells'])

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
spikemon_input = SpikeMonitor(ff_cells, name='input_spk')

# Training
Net.add(orca, ff_cells)
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

last_sequence_t = training_duration - int(sequence_duration/2/defaultclock.dt)*defaultclock.dt*3
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
