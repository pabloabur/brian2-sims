import numpy as np
from scipy.stats import norm
import brian2tools.mdexport
import pprint
import copy

from brian2 import Hz, pA, nA, mA, uA, ms, mV, prefs, SpikeMonitor, StateMonitor, defaultclock,\
    SpikeGeneratorGroup, TimedArray,\
    PopulationRateMonitor, run, uS,  uF, set_device, device, get_device

from teili import TeiliNetwork
from teili.stimuli.testbench import SequenceTestbench
from teili.tools.group_tools import add_group_activity_proxy
from teili.models.builder.synapse_equation_builder import SynapseEquationBuilder
from teili.models.builder.neuron_equation_builder import NeuronEquationBuilder

static_synapse_model = SynapseEquationBuilder(base_unit='quantized',
    plasticity='non_plastic')
neuron_model = NeuronEquationBuilder(base_unit='quantized',
    position='spatial')
from equations.neurons.LIF import LIF
from equations.synapses.CUBA import CUBA
from equations.synapses.iSTDP import iSTDP
from builder.groups_builder import create_synapses, create_neurons

import sys
import pickle
import os
from datetime import datetime

#############
# Load models
path = os.path.expanduser('/Users/Pablo/git/teili/')
model_path = os.path.join(path, "teili", "models", "equations", "")
#adp_synapse_model = SynapseEquationBuilder.import_eq(
#        model_path + 'StochInhStdp.py')
static_synapse_model = CUBA()
neuron_model = LIF()
adp_synapse_model = iSTDP()

# Initialize simulation preferences
prefs.codegen.target = "numpy"
#set_device('markdown', filename='model_description')
defaultclock.dt = 1 * ms

#################
# Building network
# cells
num_exc = 8000
num_inh = 2000

neuron_model.refractory = '5*ms'
old_pattern = 'gtot = gtot0 : volt'
new_pattern = 'gtot = gtot0 + gtot1 + gtot2 : volt'
neuron_model.modify_model('model', new_pattern, old_pattern)
neuron_model.model += 'gtot1 : volt\n' + 'gtot2 : volt\n'
Iconst = 201*pA
neuron_model.parameters['Iconst'] = Iconst
exc_cells = create_neurons(num_exc, neuron_model)

inh_cells = create_neurons(num_inh, neuron_model)

# Register proxy arrays
dummy_unit = 1*mV
exc_cells.variables.add_array('activity_proxy', 
                               size=exc_cells.N,
                               dimensions=dummy_unit.dim)
exc_cells.variables.add_array('normalized_activity_proxy', 
                               size=exc_cells.N)

# Connections
# Scales weights to maximum target
static_synapse_model.connection['p'] = .02
static_synapse_model.parameters['weight'] = '2*1/N_incoming*mV'
exc_inh_conn = create_synapses(exc_cells, inh_cells, static_synapse_model)
exc_exc_conn = create_synapses(exc_cells, exc_cells, static_synapse_model)

static_synapse_model = CUBA()
static_synapse_model.connection['p'] = .02
static_synapse_model.parameters['weight'] = '10*2*1/N_incoming*mV'
old_pattern = 'gtot0_post = g'
new_pattern = 'gtot1_post = g'
static_synapse_model.modify_model('model', new_pattern, old_pattern)
static_synapse_model.namespace['w_factor'] = -1
static_synapse_model.parameters['tau_syn'] = 10*ms
inh_inh_conn = create_synapses(inh_cells, inh_cells, static_synapse_model)

old_pattern = 'gtot0_post = g'
new_pattern = 'gtot1_post = g'
adp_synapse_model.modify_model('model', new_pattern, old_pattern)
adp_synapse_model.parameters['w_plast'] = .2e-10*mV
adp_synapse_model.parameters['tau_syn'] = 10*ms
adp_synapse_model.connection['p'] = .02
inh_exc_conn = create_synapses(inh_cells, exc_cells, adp_synapse_model)

# Time constants, stochastic
#inh_exc_conn.tausyn = 10*ms
#exc_inh_conn.tausyn = 5*ms
#exc_exc_conn.tausyn = 5*ms
#inh_inh_conn.tausyn = 10*ms
#exc_cells.tau = 19*ms
#inh_cells.tau = 10*ms

#exc_cells.Vm = 3*mV
#exc_cells.I_min = -16*mA
#exc_cells.I_max = 15*mA
#inh_cells.Vm = 3*mV
#inh_cells.I_min = -16*mA
#inh_cells.I_max = 15*mA

#inh_exc_conn.weight = -1
#inh_exc_conn.w_plast = 1
#inh_exc_conn.w_max = 15
#inh_exc_conn.A_max = 15
#inh_exc_conn.taupre = 20*ms
#inh_exc_conn.taupost = 20*ms
#inh_exc_conn.rand_num_bits_syn = 4
#inh_exc_conn.stdp_thres = 3
#inh_inh_conn.weight = -1
#exc_inh_conn.weight = 1
#exc_exc_conn.weight = 1

# Add proxy activity group
#activity_proxy_group = [exc_cells]
#add_group_activity_proxy(activity_proxy_group,
#                         buffer_size=300,
#                         decay=150)
#variance_th = 13
#inh_exc_conn.variance_th = np.random.uniform(
#        low=variance_th - 0.1,
#        high=variance_th + 0.1,
#        size=len(inh_exc_conn))
inh_exc_conn.namespace['eta'] = 0*mV

##################
# Setting up monitors
spikemon_exc_neurons = SpikeMonitor(exc_cells, name='spikemon_exc_neurons')
spikemon_inh_neurons = SpikeMonitor(inh_cells, name='spikemon_inh_neurons')
#statemon_ee_conns = StateMonitor(exc_exc_conn, variables=['g'], record=True,
#                                  name='statemon_ee_conns')
#statemon_ie_conns = StateMonitor(inh_exc_conn, variables=['g'], record=True,
#                                  name='statemon_ie_conns')
#statemon_inh_conns = StateMonitor(inh_exc_conn, variables=['w_plast', 'normalized_activity_proxy'], record=True,
#                                  name='statemon_inh_conns')
#statemon_inh_cells = StateMonitor(inh_cells, variables=['Vm'], record=True,
#                                  name='statemon_inh_cells')
#statemon_exc_cells = StateMonitor(exc_cells, variables=['Vm', 'normalized_activity_proxy', 'gtot'], record=True,
#                                  name='statemon_exc_cells')
#statemon_pop_rate_e = PopulationRateMonitor(exc_cells)

run(1000*ms, report='stdout', report_period=100*ms, namespace={})
#variance_th = 2
#inh_exc_conn.variance_th = np.random.uniform(
#        low=variance_th - 0.1,
#        high=variance_th + 0.1,
#        size=len(inh_exc_conn))
inh_exc_conn.namespace['eta'] = 1*mV
if get_device().__class__.__name__ != 'MdExporter':
    run(2000*ms, report='stdout', report_period=100*ms, namespace={})

# Plots
if get_device().__class__.__name__ != 'MdExporter':
    from brian2 import *
    figure()
    _ = hist(inh_exc_conn.w_plast/mV, bins=20)
    xlabel('Inh. weight')
    ylabel('Count')
    title('Distribution of inhibitory weights')

    #figure()
    #y = np.mean(statemon_exc_cells.normalized_activity_proxy, axis=0)
    #stdd=np.std(statemon_exc_cells.normalized_activity_proxy, axis=0)
    #plot(statemon_inh_conns.t/ms, y)
    #ylabel('normalized activity mean value')
    #xlabel('time (ms)')
    #ylim([-0.05, 1.05])
    #fill_between(statemon_inh_conns.t/ms, y-stdd, y+stdd, facecolor='lightblue')
    #annotate(f"""Shown for the sake of comparison with ADP
    #           variance threshold: 14 for first 10s, 2 afterwards""", xy = (0, 0.1))
    #
    #figure()
    #plot(statemon_exc_cells.normalized_activity_proxy.T)
    #xlabel('time (ms)')
    #ylabel('normalized activity value')
    #title('Normalized activity of all neurons')

    figure()
    plot(np.array(statemon_pop_rate_e.t/ms), np.array(statemon_pop_rate_e.smooth_rate(width=60*ms)/Hz))
    xlabel('time (ms)')
    ylabel('Mean firing rate (Hz)')
    title('Effect of inhibitory weights on firing pattern of excitatory neurons')

    tot_i_curr = np.sum(statemon_ie_conns.g, axis=0)
    figure()
    #plot(statemon_ie_conns.t/ms, -tot_i_curr/amp, color='b', label='summed inh. current')
    plot(statemon_exc_cells.t/ms,
         10*nS*statemon_exc_cells[0].gtot/amp + Iconst/amp,
         color='k', label='total current on neuron 0')
    ylabel('Current')
    xlabel('time')
    legend()

    figure()
    plot(spikemon_exc_neurons.t/ms, spikemon_exc_neurons.i, 'k.', ms=.25)
    title('Exc. neurons')
    figure()
    plot(spikemon_inh_neurons.t/ms, spikemon_inh_neurons.i, 'k.', ms=.25)
    title('Inh. neurons')

    # TODO vectorize by working with flattened array
    #win_slidings = []
    #win_size = 30
    ## Get indices to make calculations over a sliding window
    #for y in range(np.shape(statemon_exc_cells.Vm)[1]):
    #    win_slidings.append([])
    #    for x in range(win_size):
    #            win_slidings[-1].append(x+y)
    #
    # Adjust last win_size-1 intervals which are outside array dimensions
    #trim_win = [val[:-(i+1)] for i, val in enumerate(win_slidings[-win_size+1:])]
    #win_slidings[-win_size+1:] = trim_win
    #
    #mean_exc_rate = []
    #mean_inh_rate = []
    #for i in win_slidings:
    #    mean_exc_rate.append(np.mean(statemon_exc_cells.Vm[:,i]))
    #    mean_inh_rate.append(np.mean(statemon_inh_cells.Vm[:,i]))
    #
    #figure()
    #plot(mean_exc_rate)
    #title('Mean of 100 excitatory neurons')
    #xlabel('time(ms)')
    #ylabel('V')
    #
    #figure()
    #plot(mean_inh_rate)
    #title('Mean of 25 inhibitory neurons')
    #xlabel('time(ms)')
    #ylabel('V')

    show()
