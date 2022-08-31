import numpy as np

from brian2 import run, prefs, defaultclock, ms, Hz, set_device,\
        PoissonGroup, SpikeMonitor, StateMonitor, mV

import matplotlib.pyplot as plt
from brian2tools import plot_synapses, plot_state, brian_plot
from brian2tools.plotting.synapses import _float_connection_matrix
from teili.tools.sorting import SortMatrix

from equations.neurons.LIF import LIF
from equations.synapses.CUBA import CUBA
from equations.synapses.hSTDP import hSTDP
from builder.groups_builder import create_synapses, create_neurons

#prefs.codegen.target = "numpy"
#set_device('cpp_standalone')
defaultclock.dt = 1 * ms

num_neurons = 10
poisson_spikes = PoissonGroup(num_neurons, 30*Hz)

neu_model = LIF()
syn_model = CUBA()
hSTDP_model = hSTDP()

neu_model.model += 'incoming_weights : volt\n' + 'outgoing_weights : volt\n'
old_pattern = 'gtot = gtot0 : volt'
new_pattern = 'gtot = gtot0 + gtot1 : volt'
neu_model.modify_model('model', new_pattern, old_pattern)
neu_model.model += 'gtot1 : volt\n'
neurons = create_neurons(num_neurons, neu_model)

syn_model.parameters['weight'] = 80*mV
syn_model.connection['j'] = 'i'
input_syn = create_synapses(poisson_spikes, neurons, syn_model)

old_pattern = 'gtot0_post = g'
new_pattern = 'gtot1_post = g'
hSTDP_model.modify_model('model', new_pattern, old_pattern)
hSTDP_model.connection['condition'] = 'i != j'
m_fator = 1
hSTDP_model.namespace['w_max'] = 35*mV
hSTDP_model.namespace['w_lim'] = m_fator * hSTDP_model.namespace['w_max']
hSTDP_model.parameters['w_plast'] = hSTDP_model.namespace['w_max']/num_neurons/10
rec_conns = create_synapses(neurons, neurons, hSTDP_model)
# Update comes here because there is an error if declared in model definition
rec_conns.run_regularly('w_plast = clip(w_plast - h_eta*heterosyn_factor, 0*volt, w_max)',
                        dt=1*ms)

spikemon_neu = SpikeMonitor(neurons)
statemon_syn = StateMonitor(rec_conns, variables=['w_plast'],
                            record=[x for x in range(num_neurons*(num_neurons-1))])
statemon_neu = StateMonitor(neurons,
                            variables=['Vm', 'outgoing_weights', 'incoming_weights'],
                            record=[x for x in range(num_neurons)])

sim_t = 40000*ms
sim_slice = sim_t/ms-1000
run(sim_t)

# TODO assemblies
from neo.core import SpikeTrain
import quantities as nq
from elephant.spade import spade
import viziphant
spike_trains = []
for vals in spikemon_neu.spike_trains().values():
    aux_train = [spk_t for spk_t in vals/ms if spk_t>sim_slice]
    spike_trains.append(SpikeTrain(aux_train*nq.ms,
                        t_start=sim_slice*nq.ms,
                        t_stop=(sim_t/ms)*nq.ms))
patterns = spade(spike_trains, bin_size=2 * nq.ms, winlen=10)['patterns']
axes = viziphant.patterns.plot_patterns(spike_trains, patterns[:3])

plt.figure()
brian_plot(spikemon_neu)
plt.xlim(sim_slice, sim_t/ms)
plt.figure()
plot_synapses(rec_conns.i, rec_conns.j, rec_conns.w_plast, plot_type='image')
plt.figure()
mat = _float_connection_matrix(rec_conns.i, rec_conns.j, rec_conns.w_plast)
s_mat = SortMatrix(num_neurons, rec_matrix=True, matrix=mat)
plt.imshow(s_mat.sorted_matrix)
plt.title('sorted matrix')

plt.figure()
brian_plot(statemon_syn)
plt.title('All weights evolution')

plt.figure()
plot_state(statemon_neu.t, statemon_neu.outgoing_weights.T, var_name='w_plast')
plt.title('Outgoing weights sum per neuron')

plt.figure()
plot_state(statemon_neu.t, statemon_neu.incoming_weights.T, var_name='w_plast')
plt.title('Incoming weights sum per neuron')

plt.figure()
plot_state(statemon_syn.t, statemon_syn.w_plast[rec_conns.j==0, :].T, var_name='w_plast')
plt.title('incoming weights to neuron 0')

fig, axs = plt.subplots(5, 2)
axs = axs.flatten()
for n in range(num_neurons):
    plot_state(statemon_neu.t, statemon_neu.incoming_weights[n].T,
               var_name='w_plast', label='neuron variable', axes=axs[n])
    plot_state(statemon_syn.t,
               np.sum(statemon_syn.w_plast[rec_conns.j==n, :].T, axis=1),
               var_name='w_plast', label='sum of weights', axes=axs[n])
fig.suptitle('Interval neuronal variable vs sum of synaptic variables (incoming)')
plt.legend()

fig2, axs2 = plt.subplots(5, 2)
axs2 = axs2.flatten()
for n in range(num_neurons):
    plot_state(statemon_neu.t, statemon_neu.outgoing_weights[n].T,
               var_name='w_plast', label='neuron variable', axes=axs2[n])
    plot_state(statemon_syn.t,
               np.sum(statemon_syn.w_plast[rec_conns.i==n, :].T, axis=1),
               var_name='w_plast', label='sum of weights', axes=axs2[n])
fig2.suptitle('Interval neuronal variable vs sum of synaptic variables (outgoing)')
plt.legend()

plt.show()
