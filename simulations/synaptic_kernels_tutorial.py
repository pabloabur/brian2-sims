import sys
import numpy as np

from brian2tools import plot_state, brian_plot
import matplotlib.pyplot as plt

from brian2 import second, ms, mV, mA, prefs, run,\
    SpikeMonitor, StateMonitor,\
    SpikeGeneratorGroup, ExplicitStateUpdater, defaultclock
from teili.tools.misc import DEFAULT_FUNCTIONS, minifloat2decimal,\
        decimal2minifloat
from builder.groups_builder import create_synapses, create_neurons
from equations.neurons.fp8LIF import fp8LIF
from equations.synapses.fp8CUBA import fp8CUBA
from equations.neurons.LIF import LIF
from equations.synapses.CUBA import CUBA

prefs.codegen.target = "numpy"
defaultclock.dt = 1*ms

input_timestamps = np.asarray([1, 3, 6, 9, 12, 15, 18, 21]) * ms
input_indices = np.asarray([0, 1, 0, 1, 0, 1, 0, 1])

input_spikegenerator = SpikeGeneratorGroup(2, indices=input_indices,
                                           times=input_timestamps, name='gtestInp1')

neu_model_fp8 = fp8LIF()
fp8_neurons = create_neurons(1, neu_model_fp8)
neu_model = LIF()
std_neurons = create_neurons(1, neu_model)

syn_model_fp8 = fp8CUBA()
fp8_synapse = create_synapses(input_spikegenerator, fp8_neurons, syn_model_fp8)
fp8_synapse.weight[1] = decimal2minifloat(-10)

syn_model = CUBA()
std_synapse = create_synapses(input_spikegenerator, std_neurons, syn_model)
std_synapse.weight[1] = -1*mV

# Set monitors
spikemon_inp = SpikeMonitor(input_spikegenerator, name='spikemon_inp')

statemon_fp8 = StateMonitor(fp8_neurons, variables=['gtot'], record=0)
statemon_std = StateMonitor(std_neurons, variables=['gtot'], record=0)

duration = 0.040 * second
run(duration)

brian_plot(spikemon_inp)
plt.figure()
plot_state(statemon_fp8.t, minifloat2decimal(statemon_fp8.gtot[0]),
           var_name='postsynaptic current')
plt.title('fp8 model')

plt.figure()
brian_plot(statemon_std)
plt.title('standard model')

plt.show()
