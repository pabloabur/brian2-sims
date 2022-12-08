import sys
import numpy as np

from brian2tools import plot_state, brian_plot
import matplotlib.pyplot as plt

from brian2 import second, ms, mV, mA, prefs, run,\
    SpikeMonitor, StateMonitor, set_device,\
    SpikeGeneratorGroup, ExplicitStateUpdater, defaultclock
from core.utils.misc import DEFAULT_FUNCTIONS, minifloat2decimal,\
        decimal2minifloat
from core.builder.groups_builder import create_synapses, create_neurons
from core.equations.neurons.fp8LIF import fp8LIF
from core.equations.synapses.fp8CUBA import fp8CUBA
from core.equations.neurons.int4LIF import int4LIF
from core.equations.synapses.int4CUBA import int4CUBA
from core.equations.neurons.int8LIF import int8LIF
from core.equations.synapses.int8CUBA import int8CUBA
from core.equations.neurons.LIF import LIF
from core.equations.synapses.CUBA import CUBA

prefs.codegen.target = "numpy"
#set_device('cpp_standalone')
defaultclock.dt = 1*ms

input_timestamps = np.asarray([1, 3, 6, 9, 12, 15, 18, 21]) * ms
input_indices = np.asarray([0, 1, 0, 1, 0, 1, 0, 1])

input_spikegenerator = SpikeGeneratorGroup(2, indices=input_indices,
                                           times=input_timestamps, name='gtestInp1')

neu_model_fp8 = fp8LIF()
fp8_neurons = create_neurons(1, neu_model_fp8)
neu_model_int8 = int8LIF()
int8_neurons = create_neurons(1, neu_model_int8)
neu_model_int4 = int4LIF()
int4_neurons = create_neurons(1, neu_model_int4)
neu_model = LIF()
std_neurons = create_neurons(1, neu_model)

syn_model_fp8 = fp8CUBA()
fp8_synapse = create_synapses(input_spikegenerator, fp8_neurons, syn_model_fp8)
fp8_synapse.weight[1] = decimal2minifloat(-10)

syn_model_int4 = int4CUBA()
int4_synapse = create_synapses(input_spikegenerator, int4_neurons, syn_model_int4)
int4_synapse.weight[1] = -4

syn_model_int8 = int8CUBA()
int8_synapse = create_synapses(input_spikegenerator, int8_neurons, syn_model_int8)
int8_synapse.weight[1] = -64

syn_model = CUBA()
std_synapse = create_synapses(input_spikegenerator, std_neurons, syn_model)
std_synapse.weight[1] = -1*mV

# Set monitors
spikemon_inp = SpikeMonitor(input_spikegenerator, name='spikemon_inp')

statemon_fp8 = StateMonitor(fp8_neurons, variables=['gtot'], record=0)
statemon_int4 = StateMonitor(int4_neurons, variables=['gtot'], record=0)
statemon_int8 = StateMonitor(int8_neurons, variables=['gtot'], record=0)
statemon_std = StateMonitor(std_neurons, variables=['gtot'], record=0)

duration = 0.040 * second
run(duration)

brian_plot(spikemon_inp)
plt.figure()
plot_state(statemon_fp8.t, minifloat2decimal(statemon_fp8.gtot[0]),
           var_name='postsynaptic current')
plt.title('fp8 model')

plt.figure()
brian_plot(statemon_int4)
plt.title('int4 model')

plt.figure()
brian_plot(statemon_int8)
plt.title('int8 model')

plt.figure()
brian_plot(statemon_std)
plt.title('standard model')

plt.show()
