from teili import TeiliNetwork
from teili.core.groups import Neurons, Connections
from teili.models.builder.neuron_equation_builder import NeuronEquationBuilder
from teili.models.builder.synapse_equation_builder import SynapseEquationBuilder
from teili.tools.misc import neuron_group_from_spikes
from teili.models.neuron_models import QuantStochLIF as teili_neu
from teili.models.synapse_models import QuantStochSynStdp as teili_syn
from teili.models.synapse_models import QuantStochSyn as teili_static_syn

from brian2 import Hz, mV, mA, second, ms, prefs, SpikeMonitor, StateMonitor,\
        defaultclock, PoissonGroup, ExplicitStateUpdater, Clock

import os
import matplotlib.pyplot as plt
import numpy as np

mode = 't' # Teili, or Song

sim_duration = 150*second
if mode == 't':
    defaultclock.dt = 1 * ms

# Preparing input
N = 1000
F = 15*Hz
input_group = PoissonGroup(N, rates=F)

net = TeiliNetwork()
temp_monitor = SpikeMonitor(input_group, name='temp_monitor')
net.add(input_group, temp_monitor)
print('Converting Poisson input into neuro group...')
net.run(sim_duration, report='text')
input_group = neuron_group_from_spikes(
    N, defaultclock.dt, sim_duration,
    spike_indices=np.array(temp_monitor.i),
    spike_times=np.array(temp_monitor.t)*second)
del temp_monitor

# Loading models
path = os.path.expanduser("~")
model_path = os.path.join(path, 'git', 'teili', 'tutorials')
song_neu = NeuronEquationBuilder.import_eq(model_path+'/song_neu.py')
song_syn = SynapseEquationBuilder.import_eq(model_path+'/song_syn.py')

if mode == 't':
    neurons = Neurons(N=1,
                      method=ExplicitStateUpdater('''x_new = f(x,t)'''),
                      equation_builder=teili_neu(num_inputs=2))
    S = Connections(input_group, neurons,
                    method=ExplicitStateUpdater('''x_new = f(x,t)'''),
                    equation_builder=teili_syn)
    #periodic_input = PoissonGroup(1, rates=20*Hz)
    #extra_S = Connections(periodic_input, neurons,
    #                      method=ExplicitStateUpdater('''x_new = f(x,t)'''),
    #                      equation_builder=teili_static_syn)

else:
    neurons = Neurons(N=1, equation_builder=song_neu)
    S = Connections(input_group, neurons, equation_builder=song_syn)

# Initializations
S.connect()
#extra_S.connect()
S.w_plast = np.random.rand(len(S.w_plast)) * S.w_max
#extra_S.weight = 100
mon = StateMonitor(S, ['w_plast', 'Apre', 'Apost'],
                   record=[0, 1],
                   clock=Clock(2*second))
neu_mon = SpikeMonitor(neurons)


if mode == 't':
    #neurons.Iconst = 15.*mA # Force ca. 25Hz when no other input is provided
    S.taupre = 20*ms
    S.taupost = 30*ms # 30 for more depotentiation
    S.w_max = 15
    S.dApre = 15
    S.rand_num_bits_Apre = 4
    S.rand_num_bits_Apost = 4 # or 5
    S.stdp_thres = 1
    S.weight = 1
    S.gain_syn = (1/32)*mA #Needed for high N and rate

net = TeiliNetwork()
net.add(neurons, neu_mon, input_group, S, mon)#, periodic_input, extra_S)
net.run(sim_duration, report='text')

plt.subplot(311)
plt.hist(S.w_plast / S.w_max, 20)
plt.xlabel('Weight / w_max')
plt.subplot(312)
plt.plot(mon.t/second, mon.w_plast.T/S.w_max[0])
plt.xlabel('Time (s)')
plt.ylabel('Weight / w_max')
plt.subplot(313)
plt.plot(neu_mon.t/ms/1000, neu_mon.i, '.k')
plt.xlabel('Time (s)')
plt.ylabel('spikes')
plt.tight_layout()

plt.figure()
plt.plot(mon.Apre[0], 'r')
plt.plot(mon.Apre[1], 'b')

plt.figure()
plt.plot(mon.Apost[0], 'b')
plt.plot(mon.Apost[1], 'b')
plt.title('post- time windows')

plt.show()
