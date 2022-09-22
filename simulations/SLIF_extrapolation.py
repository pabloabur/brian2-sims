import numpy as np
import matplotlib.pyplot as plt

#from parameters.orca_params import ConnectionDescriptor, PopulationDescriptor
from plots.plot_utils import raster_sort
from core.utils.SLIF_utils import neuron_rate, permutation_from_rate
#from orca_column import orcaColumn
from core.parameters.monitor_params import monitor_params, selected_cells

from brian2tools.plotting.synapses import _float_connection_matrix
from brian2tools import plot_state
import brian2tools.mdexport
from brian2tools import brian_plot

from brian2 import ms, mA, mV, second, prefs, SpikeMonitor, StateMonitor,\
        defaultclock, ExplicitStateUpdater, set_device, device,\
        Network, collect, get_device, profiling_summary

from core.utils.testbench import SequenceTestbench
from core.utils.misc import neuron_group_from_spikes

import sys
import pickle
import json
import os
from datetime import datetime
import argparse


def change_params_conn1(desc):
    # Changes some intralaminar connections
    desc.plasticities['sst_pv'] = 'static'
    desc.plasticities['pyr_pyr'] = 'hredsymstdp'
    desc.probabilities['pyr_pyr'] = 1.0
    desc.filter_params()

    desc.base_vals['pyr_pyr']['stdp_thres'] = 2


def change_params_conn2(desc):
    # Changes input parameters
    desc.plasticities['ff_pyr'] = 'hredsymstdp'
    desc.filter_params()

    desc.base_vals['ff_pyr']['stdp_thres'] = 2


def change_params_pop1(desc):
    # Changes proportions of the EI population
    desc.e_ratio = .00224
    desc.filter_params()


def testing_code(orca):
    orca.groups['pyr_pyr'].stdp_thres = 0
    orca.groups['pyr_cells'].tau_thr = 3600*second
    orca.groups['sst_pyr'].stdp_thres = 0
    orca.groups['pv_pyr'].stdp_thres = 0
    orca.groups['pv_cells'].tau_thr = 3600*second

# Initialize simulation preferences
# Use numpy
prefs.codegen.target = "numpy"
# Or faster C++
#set_device('cpp_standalone', directory='output', build_on_run=False)
#set_device('markdown', filename='model_description')
rng = np.random.default_rng(12345)
defaultclock.dt = 1*ms

# Initialize input sequence
num_items = 9
item_duration = 20
num_channels = 36
noise_prob = None
item_rate = 128
repetitions = 200

parser = argparse.ArgumentParser(description=f'Tests extrapolations on '
                                             f'different scenarios')
parser.add_argument('--gaps', type=int, default=0,
                    help='time gaps between symbols, in ms')
parser.add_argument('--path', default=None, help='directory to save')
parser.add_argument('--trial', type=int, default=0, help='trial number')
parser.add_argument('--quiet', action='store_true',
                    help='whether to create plots/info or not')
parser.add_argument('--miss', type=int, default=None, nargs=2,
                    help='defines range of symbols to remove when testing')
args = parser.parse_args()
path = args.path
time_gap = args.gaps
quiet = args.quiet
trial_no = args.trial
missing_range = args.miss

spike_indices = []
spike_times = []
#item_superposition = time_gap
#for i in range(repetitions):
#    sequence = SequenceTestbench(num_channels, num_items, item_duration,
#                                 item_superposition, noise_prob, item_rate,
#                                 1, deterministic=True)
#    tmp_i, tmp_t = sequence.stimuli()
#    aux_num_items = num_items
#
#    # Alternative scenario: a pattern that goes back and forth
#    # Adjust superposition
#    #del_idx = [x for x, y in enumerate(tmp_i)
#    #   if y in np.unique(sequence.items[num_items-1]['i'])]
#    #tmp_i, tmp_t = np.delete(tmp_i, del_idx), np.delete(tmp_t/ms, del_idx)*ms
#    #if not i%2:
#    #    tmp_i = (num_channels-1-tmp_i)
#    #aux_num_items = num_items - 1
#
#    # Alternative scenario: a brief direction selectivity
#    #aux_num_items = num_items - 1
#    #if i>350:
#    #    tmp_i = (num_channels-1-tmp_i)
#
#    spike_indices.extend(tmp_i)
#    tmp_t = [x/ms + i*(aux_num_items*(item_duration - item_superposition)
#                       + item_superposition + time_gap)
#             for x in tmp_t]
#    spike_times.extend(tmp_t)
#input_indices = np.array(spike_indices)
#input_times = np.array(spike_times) * ms
#sequence_duration = sequence.cycle_length * ms + time_gap*num_items*ms

# This input is very specific for this case.
isi = np.ceil(1/item_rate*1000).astype(int)
input_times = []
input_indices = []
for i in range(repetitions):
    for j in range(num_items):
        try:
            last_t = input_times[-1]
        except IndexError:
            last_t = 0
        tmp_t = np.repeat([x+time_gap for x in range(last_t, (2*isi+last_t)+1, isi)], 4).tolist()
        input_times.extend(tmp_t)

        tmp_i = [x for x in range(4*j, 4*j+4)]*3
        input_indices.extend(tmp_i)
input_indices = np.array(input_indices)
input_times = np.array(input_times) * ms - time_gap*ms
aux_ind = np.where(np.cumsum(input_indices==(num_channels-1))==3)[0]
sequence_duration = input_times[aux_ind[0]]

sim_duration = np.max(input_times)
# Testing time adjusted so that training ends with complete sequence
test_samples = 25
testing_duration = test_samples * sequence_duration

# Convert input into neuron group (necessary for STDP compatibility)
relay_cells = neuron_group_from_spikes(num_channels,
                                       defaultclock.dt,
                                       sim_duration,
                                       spike_indices=input_indices,
                                       spike_times=input_times)

#column = orcaColumn(['L4'])
from core.equations.neurons.LIF import LIF
from core.equations.neurons.LIFIP import LIFIP
from core.equations.synapses.CUBA import CUBA
from core.equations.synapses.hSTDP import hSTDP
from core.equations.synapses.STDP import STDP
from core.equations.synapses.iSTDP import iSTDP
from core.builder.groups_builder import create_synapses, create_neurons
from core.parameters.orca_params import syn_intra_prob, syn_intra_plast, neu_pop,\
        syn_input_prob, syn_input_plast

# load and change models. final conn_desc now would have only existing connections
conn_desc = {key: {'connection': {}} for key in syn_intra_prob['L4'].keys()}
conn_desc = {**conn_desc,
            **{key: {'connection': {}} for key in syn_input_prob['L4'].keys()}
             }

# Getting probabilities
for key in syn_intra_prob['L4']:
    if not syn_intra_prob['L4'][key]:
        del conn_desc[key]
        continue
    conn_desc[key]['connection']['p'] = syn_intra_prob['L4'][key]
for key in syn_input_prob['L4']:
    if not syn_input_prob['L4'][key]:
        del conn_desc[key]
        continue
    conn_desc[key]['connection']['p'] = syn_input_prob['L4'][key]

# Merging dicts
for key in conn_desc:
    if key in syn_intra_plast['L4']:
        conn_desc[key]['plast'] = syn_intra_plast['L4'][key]
    if key in syn_input_plast['L4']:
        conn_desc[key]['plast'] = syn_input_plast['L4'][key]
conn_desc['pyr_pyr']['connection']['p'] = 1.0
conn_desc['pyr_pyr']['plast'] = 'hstdp'
conn_desc['sst_pv']['plast'] = 'static'
conn_desc['ff_pyr']['plast'] = 'stdp'

# Removing things not used for network
conn_desc = {key: val for key, val in conn_desc.items() if 'fb_' not in key}

# add populations
pops = neu_pop['L4']
del pops['n_exc']
del pops['num_inputs']
pops['num_pyr'] = 46
num_inh = int(pops['num_pyr']/pops['ei_ratio'])
for inh_neu, ratio in pops['inh_ratio'].items():
    inh_id = inh_neu.split('_')[0]
    pops[f'num_{inh_id}'] = int(num_inh * ratio)
pops['plast'] = {'pyr_cells': 'adapt',
                 'pv_cells': 'adapt',
                 'sst_cells': 'static',
                 'vip_cells': 'static'}

tau_m_sample = {'attr': 'parameters', 'key': 'tau_m',
                'new_expr': '(randn()*10*pF + Cm)/gl'}
refrac_period = {'attr': 'refractory', 'new_expr': '3*ms'}
inh_sign = {'attr': 'namespace', 'key': 'w_factor', 'new_expr': -1}
inh_w = {'attr': 'parameters', 'key': 'weight',
         'new_expr': 'clip(2 + randn(), 0, inf)*mV'}
plast_inh_w = {'attr': 'parameters', 'key': 'w_plast',
               'new_expr': 'clip(2 + randn(), 0, inf)*mV'}
tau_syn_sample = {'attr': 'parameters', 'key': 'tau_syn',
                  'new_expr': 'clip(5 + randn(), 0, inf)*ms'}
static_inp_w = {'attr': 'parameters', 'key': 'weight',
                'new_expr': 'clip(3 + randn(), 0, inf)*mV'}
plastic_inp_w = {'attr': 'parameters', 'key': 'w_plast',
                 'new_expr': 'clip(3 + randn(), 0, inf)*mV'}
delay_sample = {'attr': 'parameters', 'key': 'delay',
                'new_expr': 'rand()*20*ms'}
rec_w = {'attr': 'parameters', 'key': 'w_plast',
         'new_expr': 'clip(1 + randn(), 0, inf)*mV'}
inh_inp_w = {'attr': 'parameters', 'key': 'weight',
             'new_expr': 'clip(3 + randn(), 0, inf)*mV'}
sample_itrace = {'attr': 'parameters', 'key': 'tau_itrace',
                 'new_expr': 'clip(20 + 2*randn(), 0, inf)*ms'}
sample_jtrace = {'attr': 'parameters', 'key': 'tau_jtrace',
                 'new_expr': 'clip(30 + 2*randn(), 0, inf)*ms'}
syn_competition = {'attr': 'namespace', 'key': 'w_lim', 'new_expr': 100*mV}
max_we = {'attr': 'namespace', 'key': 'w_max', 'new_expr': 35*mV}

params_modifier = {
    # cells
    'pyr_cells': [tau_m_sample, refrac_period],
    'pv_cells': [tau_m_sample, refrac_period],
    'sst_cells': [tau_m_sample, refrac_period],
    'vip_cells': [tau_m_sample, refrac_period],
    # inhibitory connections
    'pv_pyr': [inh_sign, tau_syn_sample, plast_inh_w, sample_itrace, sample_jtrace],
    'pv_pv': [inh_sign, tau_syn_sample, inh_w],
    'sst_pyr': [inh_sign, tau_syn_sample, plast_inh_w, sample_itrace, sample_jtrace],
    'sst_pv': [inh_sign, tau_syn_sample, inh_w],
    'sst_vip': [inh_sign, tau_syn_sample, inh_w],
    'vip_sst': [inh_sign, tau_syn_sample, inh_w],
    # excitatory connections
    'ff_pyr' : [tau_syn_sample, plastic_inp_w, sample_itrace, sample_jtrace],
    'ff_pv' : [tau_syn_sample, static_inp_w],
    'ff_sst' : [tau_syn_sample, static_inp_w],
    'ff_vip' : [tau_syn_sample, static_inp_w],
    'pyr_pyr' : [tau_syn_sample, delay_sample, rec_w, sample_itrace,
                 sample_jtrace, syn_competition, max_we],
    'pyr_pv' : [tau_syn_sample, inh_inp_w],
    'pyr_sst' : [tau_syn_sample, inh_inp_w],
    'pyr_vip' : [tau_syn_sample, inh_inp_w],
    }

column = {}
column['ff_cells'] = relay_cells
for neu_group, plast in pops['plast'].items():
    target = neu_group.split('_')[0]
    if plast == 'static':
        neu_model = LIF()
    elif plast == 'adapt':
        neu_model = LIFIP()

    # Process models according to connections, one target cell at a time
    input_sources = []
    for conn in conn_desc.keys():
        tmp_source, temp_target = conn.split('_')[0], conn.split('_')[1]
        if temp_target != target:
            continue
        input_sources.append(tmp_source)
    old_pattern = 'gtot = gtot0'
    new_pattern = 'gtot = ' + ' + '.join(['gtot' + s for s in input_sources])
    neu_model.modify_model('model', new_pattern, old_pattern)
    for in_source in input_sources:
        neu_model.model += ('\ngtot' + in_source + ' : volt')
        if conn_desc[in_source + '_' + target]['plast'] == 'hstdp':
            neu_model.model += ('\nincoming_weights : volt' + '\noutgoing_weights : volt')

    if neu_group in params_modifier:
        for modifier in params_modifier[neu_group]:
            neu_model.modify_model(**modifier)
    column[neu_group] = create_neurons(pops[f'num_{target}'], neu_model)

# add connections
conn_desc['pyr_pv']['connection']['p'] = 0.79
conn_desc['pyr_sst']['connection']['p'] = 0.79
conn_desc['pyr_vip']['connection']['p'] = 0.79
for conn, conn_vals in conn_desc.items():
    source, target = conn.split('_')[0], conn.split('_')[1]
    if conn_vals['plast'] == 'static':
        syn_model = CUBA()
    elif conn_vals['plast'] == 'stdp':
        syn_model = STDP()
    elif conn_vals['plast'] == 'istdp':
        syn_model = iSTDP()
    elif conn_vals['plast'] == 'hstdp':
        syn_model = hSTDP()

    for key, vals in conn_vals['connection'].items():
        syn_model.connection[key] = vals
    syn_model.modify_model('model', f'gtot{source}_post', old_expr='gtot0_post')
    if conn in params_modifier:
        for modifier in params_modifier[conn]:
            syn_model.modify_model(**modifier)
    column[conn] = create_synapses(column[f'{source}_cells'],
                                   column[f'{target}_cells'],
                                   syn_model)

    if conn_vals['plast'] == 'hstdp':
        column[conn].run_regularly('w_plast = clip(w_plast - h_eta*heterosyn_factor, 0*volt, w_max)',
                                   dt=1*ms)

#conn_modifier = {'L4': change_params_conn1}
#pop_modifier = {'L4': change_params_pop1}
#column.create_layers(pop_modifier, conn_modifier)
#conn_modifier = {'L4': change_params_conn2}
#column.connect_inputs(relay_cells, 'ff', conn_modifier)

# Create readouts
#stochastic_decay = ExplicitStateUpdater('''x_new = f(x,t)''')
#pop_desc = PopulationDescriptor('L4')
#pop_desc.group_plast['pyr_cells'] = 'static'
#pop_desc.filter_params()
#pop_desc.base_vals['pyr_cells']['I_min'] = -32*mA
#pop_desc.base_vals['pyr_cells']['I_max'] = 32*mA
neu_model = LIF()
neu_model.modify_model('model', 'gtot = gtot0 + gtot1 + gtot2', old_expr='gtot = gtot0')
neu_model.model += 'gtot1 : volt\ngtot2 : volt\n'
neu_model.model += '\nincoming_weightsO : volt'
readout = create_neurons(num_items, neu_model, name='readout')

neu_model = LIF()
neu_model.modify_model('model', 'gtot = gtot0 + gtot1', old_expr='gtot = gtot0')
neu_model.model += 'gtot1 : volt\n'
inh_readout = create_neurons(num_items, neu_model, name='inh_readout')
#readout = Neurons(
#    num_items,
#    equation_builder=pop_desc.models['static'](num_inputs=2),
#    method=stochastic_decay,
#    name='readout_cells',
#    verbose=True)
#readout.set_params(pop_desc.base_vals['pyr_cells'])
#readout.refrac_decay_numerator = 240

syn_model = CUBA()
#conn_desc = ConnectionDescriptor('L4', 'input')
#conn_desc.filter_params()
syn_model.modify_model('connection', [x for x in range(num_channels)], key='i')
syn_model.modify_model('connection', np.repeat([x for x in range(num_items)], 4), key='j')
syn_model.modify_model('parameters', 18*mV, key='weight')
input_readout = create_synapses(column['ff_cells'], readout, syn_model, name='input_readout')
input_inhreadout = create_synapses(column['ff_cells'], inh_readout, syn_model, name='input_inhreadout')
input_readout.active = False
input_inhreadout.active = False

syn_model = CUBA()
syn_model.modify_model('connection', 'i', key='j')
syn_model.modify_model('model', 'gtot2_post', old_expr='gtot0_post')
syn_model.modify_model('parameters', 18*mV, key='weight')
syn_model.modify_model('namespace', -1, key='w_factor')
inhreadout_readout = create_synapses(inh_readout, readout, syn_model, name='inhreadout_readout')

syn_model = CUBA()
syn_model.modify_model('connection', 'i', key='j')
syn_model.modify_model('model', 'gtot1_post', old_expr='gtot0_post')
syn_model.modify_model('parameters', 18*mV, key='weight')
readout_inhreadout = create_synapses(readout, inh_readout, syn_model, name='readout_inhreadout')

#input_readout = Connections(
#    relay_cells, readout,
#    equation_builder=conn_desc.models['static'](),
#    method=stochastic_decay,
#    name='input_readout'
#    )
#input_readout.connect(i=[x for x in range(num_channels)],
#                      j=np.repeat([x for x in range(num_items)], 4))
#input_readout.set_params(conn_desc.base_vals['ff_pyr'])
#input_readout.weight = 9

syn_model = hSTDP()
syn_model.modify_model('model', 'gtot1_post', old_expr='gtot0_post')
syn_model.modify_model('model', '',
    old_expr='outgoing_weights_pre = w_plast : volt (summed)')
syn_model.modify_model('model', '',
    old_expr='outgoing_factor = outgoing_weights_pre - w_lim : volt')
syn_model.modify_model('model', '',
    old_expr='+ int(outgoing_factor > 0*volt)*outgoing_factor ')
syn_model.modify_model('model', 'incoming_weightsO',
    old_expr='incoming_weights')
syn_model.namespace['w_lim'] = 65*mV
syn_model.parameters['w_plast'] = 10*mV
syn_model.namespace['h_eta'] = .0005
pyr_readout = create_synapses(column['pyr_cells'], readout, syn_model, name='pyr_readout')
pyr_readout.run_regularly('w_plast = clip(w_plast - h_eta*heterosyn_factor, 0*volt, w_max)',
                           dt=1*ms)
pyr_readout.active = False

column['readout'] = readout
column['ff_readout'] = input_readout
column['ff_inhreadout'] = input_inhreadout
column['pyr_readout'] = pyr_readout

# Prepare for saving data
date_time = datetime.now()
if not path:
    path = f"""{date_time.strftime('%Y.%m.%d')}_{date_time.hour}.{date_time.minute}/"""
os.makedirs(path)

num_exc = pops['num_pyr']#column.col_groups['L4'].groups['pyr_cells'].N
if get_device().__class__.__name__ == 'CPPStandaloneDevice':
    selected_connections_input = [1, 10, 100]
    selected_connections_rec = [1, 10, 100]
else:
    selected_connections_input = True
    selected_connections_rec = True

Metadata = {'time_step': str(defaultclock.dt),
            'num_exc': num_exc,
            'num_pv': pops['num_pv'],#column.col_groups['L4'].groups['pv_cells'].N,
            'num_channels': num_channels,
            'num_items': num_items,
            'sequence_duration': str(sequence_duration),
            'repetitions': repetitions,
            'sim_duration': str(sim_duration/ms) + ' ms',
            'testing_duration': str(testing_duration/ms) + ' ms',
            'time_gap': time_gap,
            'trial_no': trial_no,
            'selected_cells': selected_cells.tolist(),
            'selected_connections_input': selected_connections_input,
            'selected_connections_rec': selected_connections_rec,
            'missing_range': missing_range,
            're_init_dt': None
            }

with open(path+'metadata.json', 'w') as f:
    json.dump(Metadata, f)

##################
# Setting up monitors
monitor_params['statemon_static_conn_ff_pyr']['group'] = 'L4_ff_pyr'
monitor_params['statemon_conn_ff_pyr']['group'] = 'L4_ff_pyr'
if get_device().__class__.__name__ == 'CPPStandaloneDevice':
    monitor_params['statemon_conn_ff_pyr']['record'] = selected_connections_input
    monitor_params['statemon_static_conn_ff_pyr']['record'] = selected_connections_input
    monitor_params['statemon_conn_pyr_pyr']['record'] = selected_connections_rec
#column.col_groups['L4'].create_monitors(monitor_params)

# Temporary monitors
spikemon_input = SpikeMonitor(relay_cells, name='input_spk')
spkmon_l4 = SpikeMonitor(column['pyr_cells'],#column.col_groups['L4']._groups['pyr_cells'],
                         name='l4_spk')
sttmon_l4 = StateMonitor(column['pyr_cells'], variables=['Vm', 'gtotff', 'gtotpyr', 'gtotsst', 'gtotpv'], record=True, name='sttmon_l4')
sttmon_vm = StateMonitor(
    readout,
    variables=['Vm', 'gtot0', 'gtot1', 'gtot2'],
    record=True,
    dt=1*ms)
sttmon_w = StateMonitor(pyr_readout, variables=['w_plast', 'g'], record=True)
spkmon_r = SpikeMonitor(readout, name='readoutmon')

#conn_desc = ConnectionDescriptor('L4', 'intra')
#conn_desc.plasticities['pyr_pyr'] = 'stdp'
#conn_desc.filter_params()
#conn_desc.base_vals['pyr_pyr']['delay'] = 0*ms
#pyr_readout = Connections(
#    column.col_groups['L4'].groups['pyr_cells'], readout,
#    equation_builder=conn_desc.models['stdp'](),
#    method=stochastic_decay,
#    name='pyr_readout'
#    )
#pyr_readout.connect()
#pyr_readout.set_params(conn_desc.base_vals['pyr_pyr'])
#pyr_readout.weight = 0

# Training
Net = Network(collect())
#Net.add([x for x in column.col_groups.values()])
Net.add(list(column.values()))
#Net.add([x.input_groups for x in column.col_groups.values()])
#Net.add(spikemon_input, spkmon_l4, sttmon_vm,
#        readout, spkmon_r, input_readout)
#Net.add(pyr_readout)

phase_one = np.floor((sim_duration - testing_duration)/2)
phase_two = (sim_duration - testing_duration) - phase_one 
Net.run(phase_one, namespace={}, profile=True)

pyr_readout.active = True
input_readout.active = True
input_inhreadout.active = True
Net.run(phase_two, namespace={}, profile=True)

if get_device().__class__.__name__ == 'MdExporter':
    import pdb;pdb.set_trace()  # better than import code

# Recover data pickled from monitor
#spikemon_exc_neurons = column.col_groups['L4'].monitors['spikemon_pyr_cells']

if testing_duration:
    # Alternative scenario: Self-sustained activity persists and eventually
    # fade away
    #column.col_groups['L4'].input_groups['L4_ff_pyr'].weight = 0
    #column.col_groups['L4'].input_groups['L4_ff_pv'].weight = 0
    #column.col_groups['L4'].input_groups['L4_ff_sst'].weight = 0
    # With new network structure:
    #column['ff_pyr'].namespace['w_factor'] = 0
    #column['ff_pv'].namespace['w_factor'] = 0
    #column['ff_sst'].namespace['w_factor'] = 0

    #input_readout.weight = 0
    #pyr_readout.weight = 1

    # Alternative scenario: missing symbols are extrapolated
    if missing_range:
        symbols = sequence.items
        rm_first_ch = np.unique(symbols[missing_range[0]]['i'])[0]
        rm_last_ch = np.unique(symbols[missing_range[1]]['i'])[-1]
        column['ff_pyr'].w_plast[f'i>={rm_first_ch} and i<={rm_last_ch}'] = 0*mV

    input_readout.namespace['w_factor'] = 0
    input_inhreadout.namespace['w_factor'] = 0
    inhreadout_readout.namespace['w_factor'] = 0
    pyr_readout.namespace['eta'] = 0*mV

    #testing_code(column.col_groups['L4'])

    Net.run(testing_duration, namespace={}, profile=True)
    if get_device().__class__.__name__ == 'CPPStandaloneDevice':
        device.build()

# Saving results
#column.col_groups['L4'].save_data(monitor_params, path)

# Calculating permutation indices from firing rates
trials = 10
last_sequence_t = sim_duration - testing_duration - int(
    sequence_duration/defaultclock.dt)*defaultclock.dt*trials
neu_rates = neuron_rate(spkmon_l4, kernel_len=10*ms,
                        kernel_var=1*ms, simulation_dt=defaultclock.dt,
                        interval=[last_sequence_t, sim_duration - testing_duration],
                        smooth=True, trials=trials)
permutation_ids = permutation_from_rate(neu_rates)

# Save data
np.savez(path+f'input_raster.npz',
         input_t=np.array(spikemon_input.t/ms),
         input_i=np.array(spikemon_input.i)
         )

np.savez(path+f'permutation.npz',
         ids=permutation_ids
         )
  
# Saves data that is used for metrics
input_spikes = [spks/ms for spks in spikemon_input.spike_trains().values()]
with open(path+'input_spikes', 'wb') as f:
    pickle.dump(input_spikes, f)
output_spikes = [spks/ms for spks in spkmon_r.spike_trains().values()]
with open(path+'output_spikes', 'wb') as f:
    pickle.dump(output_spikes, f)
rec_spikes = [spks/ms for spks in spkmon_l4.spike_trains().values()]
with open(path+'rec_spikes', 'wb') as f:
    pickle.dump(rec_spikes, f)

L4_ids = raster_sort(spkmon_l4, permutation_ids)
L4_times = spkmon_l4.t/ms
#events_model1 = EventsModel(neuron_ids=L4_ids, spike_times=L4_times)

input_ids, input_times = spikemon_input.i, spikemon_input.t/ms
#events_model3 = EventsModel(neuron_ids=input_ids, spike_times=input_times)

input_ids, input_times = spkmon_r.i, spkmon_r.t/ms
#events_model4 = EventsModel(neuron_ids=input_ids, spike_times=input_times)

#plot_settings = PlotSettings(colors=['r'])

#mainfig = pg.GraphicsWindow()
#subfig1 = mainfig.addPlot(row=0, col=0)
#subfig2 = mainfig.addPlot(row=1, col=0)
#subfig3 = mainfig.addPlot(row=0, col=1)
#subfig4 = mainfig.addPlot(row=1, col=1)
#subfig2.setXLink(subfig1)
#subfig4.setXLink(subfig1)
#
#raster_plot1 = Rasterplot(MyEventsModels=[events_model1],
#                         MyPlotSettings=plot_settings,
#                         subgroup_labels=['L4'], backend='pyqtgraph',
#                         mainfig=mainfig, subfig_rasterplot=subfig1,
#                         QtApp=QtApp, show_immediately=False)
#
#raster_plot3 = Rasterplot(MyEventsModels=[events_model3],
#                         MyPlotSettings=plot_settings,
#                         subgroup_labels=['input'], backend='pyqtgraph',
#                         mainfig=mainfig, subfig_rasterplot=subfig2,
#                         QtApp=QtApp, show_immediately=False)
#
#line_plot = Lineplot(DataModel_to_x_and_y_attr=[(sttmon_vm, ('t', 'Vm'))],
#                     MyPlotSettings=plot_settings,
#                     subgroup_labels=['membrane potential'], backend='pyqtgraph',
#                     mainfig=mainfig, subfig=subfig3,
#                     QtApp=QtApp, show_immediately=False)
#
#raster_plot4 = Rasterplot(MyEventsModels=[events_model4],
#                         MyPlotSettings=plot_settings,
#                         subgroup_labels=['readout'], backend='pyqtgraph',
#                         mainfig=mainfig, subfig_rasterplot=subfig4,
#                         QtApp=QtApp, show_immediately=True)

if not quiet:
    # regular plot of sorted weights
    #s_mat = SortMatrix(num_neurons, rec_matrix=True, matrix=mat)
    plt.figure()
    plot_state(sttmon_w.t, np.sum(sttmon_w.w_plast[pyr_readout.j==0, :].T, axis=1), var_name='w_plast')
    plt.title('weights projecting into neuron 0')

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    brian_plot(spkmon_r, axes=ax1)
    ax1.vlines((sim_duration-testing_duration)/ms, -1, 9, linestyles='dotted')
    ax2.plot(L4_times, L4_ids, '.')
    ax2.vlines((sim_duration-testing_duration)/ms, -1, 50, linestyles='dotted')
    brian_plot(spikemon_input, axes=ax3)

    rmat = _float_connection_matrix(column['pyr_pyr'].i, column['pyr_pyr'].j, column['pyr_pyr'].w_plast)
    plt.figure()
    plt.imshow(rmat[:, permutation_ids][permutation_ids, :])

    rmat = _float_connection_matrix(column['pyr_pyr'].i, column['pyr_pyr'].j, column['pyr_pyr'].w_plast)
    plt.figure()
    fmat = _float_connection_matrix(column['ff_pyr'].i, column['ff_pyr'].j, column['ff_pyr'].w_plast)
    plt.imshow(fmat.T[:, permutation_ids])
    omat = _float_connection_matrix(column['pyr_readout'].i, column['pyr_readout'].j, column['pyr_readout'].w_plast)
    plt.figure()
    plt.imshow(omat[:, permutation_ids])
    plt.show()

    profiling_summary(Net, show=10)
