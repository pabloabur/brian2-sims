#!/usr/bin/env python3
import numpy as np
import sys
from brian2 import mV, uA, mA, ms, ohm, second, pA, nA, prefs,\
    SpikeMonitor, StateMonitor, SpikeGeneratorGroup, defaultclock,\
    ExplicitStateUpdater, run, set_device, profiling_summary

from brian2tools import plot_state, brian_plot
import matplotlib.pyplot as plt

from core.utils.misc import decimal2minifloat, minifloat2decimal
from core.utils.misc import DEFAULT_FUNCTIONS

from core.equations.neurons.LIF import LIF
from core.equations.synapses.CUBA import CUBA
from core.equations.neurons.fp8LIF import fp8LIF
from core.equations.synapses.fp8CUBA import fp8CUBA
from core.equations.neurons.int4LIF import int4LIF
from core.equations.synapses.int4CUBA import int4CUBA
from core.equations.neurons.int8LIF import int8LIF
from core.equations.synapses.int8CUBA import int8CUBA
from core.builder.groups_builder import create_synapses, create_neurons

prefs.codegen.target = "numpy"
#set_device('cpp_standalone')
defaultclock.dt = 1*ms

input_timestamps = np.asarray([1, 3, 4, 5, 6, 7, 8, 9]) * ms
input_indices = np.asarray([0, 0, 0, 0, 0, 0, 0, 0])
input_spikegenerator = SpikeGeneratorGroup(1, indices=input_indices,
                                           times=input_timestamps, 
                                           name='input_spikegenerator')


# Preparing standard models
neuron_model = LIF()
test_neurons1 = create_neurons(2, neuron_model, name='std_neu1')
test_neurons2 = create_neurons(2, neuron_model, name='std_neu2')
test_neurons1.Iconst = 400*pA

synapse_model_test = CUBA()
input_synapse = create_synapses(input_spikegenerator, test_neurons1,
                                synapse_model_test, name='std_syn1')
input_synapse.weight = 80*mV
test_synapse = create_synapses(test_neurons1, test_neurons2,
                               synapse_model_test, name='std_syn2')
test_synapse.weight = 50*mV

# Preparing 8-bit models
neuron_model2 = fp8LIF()
layer1_fp8 = create_neurons(2, neuron_model2, name='fp8_neu1')
layer2_fp8 = create_neurons(2, neuron_model2, name='fp8_neu2')
layer1_fp8.Iconst = decimal2minifloat(52)

synapse_model_test2 = fp8CUBA()
input_syn_fp8 = create_synapses(input_spikegenerator, layer1_fp8,
                                 synapse_model_test2, name='fp8_syn1')
input_syn_fp8.weight = decimal2minifloat(96)
syn_fp8 = create_synapses(layer1_fp8, layer2_fp8,
                                synapse_model_test2, name='fp8_syn2')
syn_fp8.weight = decimal2minifloat(60)

neuron_model3 = int4LIF()
layer1_int4 = create_neurons(2, neuron_model3, name='int4_neu1')
layer2_int4 = create_neurons(2, neuron_model3, name='int4_neu2')
layer1_int4.Iconst = 2

neuron_model4 = int8LIF()
layer1_int8 = create_neurons(2, neuron_model4, name='int8_neu1')
layer2_int8 = create_neurons(2, neuron_model4, name='int8_neu2')
layer1_int8.Iconst = 25

synapse_model_test3 = int4CUBA()
input_syn_int4 = create_synapses(input_spikegenerator, layer1_int4,
                                 synapse_model_test3, name='int4_syn1')
input_syn_int4.weight = 7
syn_int4 = create_synapses(layer1_int4, layer2_int4,
                                synapse_model_test3, name='int4_syn2')
syn_int4.weight = 3

synapse_model_test4 = int8CUBA()
input_syn_int8 = create_synapses(input_spikegenerator, layer1_int8,
                                 synapse_model_test4, name='int8_syn1')
input_syn_int8.weight = 78
syn_int8 = create_synapses(layer1_int8, layer2_int8,
                                synapse_model_test4, name='int8_syn2')
syn_int8.weight = 42

# Monitoring standard models
spikemon_input = SpikeMonitor(input_spikegenerator)
spikemon_test_neurons2 = SpikeMonitor(test_neurons2)
statemon_input_synapse = StateMonitor(
    test_neurons1, variables='gtot', record=True)
statemon_test_synapse = StateMonitor(
    test_neurons2, variables='gtot', record=True)
statemon_test_neurons1 = StateMonitor(test_neurons1, variables=[
    'gtot', "Vm"], record=[0, 1])
statemon_test_neurons2 = StateMonitor(test_neurons2,
                                      variables=['Vm', 'gtot'],
                                      record=0)

# Monitoring 8-bit models
spikemon_layer2_fp8 = SpikeMonitor(layer2_fp8)
statemon_layer1_fp8 = StateMonitor(layer1_fp8, variables=[
    'gtot', "Vm"], record=[0, 1])
statemon_layer2_fp8 = StateMonitor(layer2_fp8,
                                      variables=['Vm', 'gtot'],
                                      record=0)

spikemon_layer2_int8 = SpikeMonitor(layer2_int8)
statemon_layer1_int8 = StateMonitor(layer1_int8, variables=[
    'gtot', "Vm"], record=[0, 1])
statemon_layer2_int8 = StateMonitor(layer2_int8,
                                      variables=['Vm', 'gtot'],
                                      record=0)

# Monitoring 4-bit models
spikemon_layer2_int4 = SpikeMonitor(layer2_int4)
statemon_layer1_int4 = StateMonitor(layer1_int4, variables=[
    'gtot', "Vm"], record=[0, 1])
statemon_layer2_int4 = StateMonitor(layer2_int4,
                                      variables=['Vm', 'gtot'],
                                      record=0)

duration = 0.5
run(duration * second, namespace={}, profile=True)

fig, axs = plt.subplots(3, 2, sharex=True)

axs[0, 0].title.set_text('input spikes')
brian_plot(spikemon_input, axes=axs[0, 0])

axs[0, 1].title.set_text('PSCs on 1st layer')
plot_state(statemon_layer1_fp8.t,
           minifloat2decimal(statemon_layer1_fp8.gtot[0]),
           var_name='gtot', axes=axs[0, 1])
plot_state(statemon_layer1_fp8.t,
           minifloat2decimal(statemon_layer1_fp8.gtot[1]),
           var_name='gtot', axes=axs[0, 1])

axs[1, 0].title.set_text('Vm on 1st layer')
plot_state(statemon_layer1_fp8.t,
           minifloat2decimal(statemon_layer1_fp8.Vm[0]),
           var_name='Vm', axes=axs[1, 0])

axs[1, 1].title.set_text('PSCs on 2nd layer')
plot_state(statemon_layer2_fp8.t,
           minifloat2decimal(statemon_layer2_fp8.gtot[0]),
           var_name='gtot', axes=axs[1, 1])

axs[2, 0].title.set_text('spikes on 2nd layer')
brian_plot(spikemon_layer2_fp8, axes=axs[2, 0])

axs[2, 1].title.set_text('Vm on 2nd layer')
plot_state(statemon_layer2_fp8.t,
           minifloat2decimal(statemon_layer2_fp8.Vm[0]),
           var_name='Vm', axes=axs[2, 1])

fig.suptitle('8-bit floating point neuron')
plt.pause(0.001)

fig, axs = plt.subplots(3, 2, sharex=True)

axs[0, 0].title.set_text('input spikes')
brian_plot(spikemon_input, axes=axs[0, 0])

axs[0, 1].title.set_text('PSCs on 1st layer')
plot_state(statemon_input_synapse.t,
           statemon_input_synapse.gtot[0],
           var_name='gtot', axes=axs[0, 1])
plot_state(statemon_input_synapse.t,
           statemon_input_synapse.gtot[1],
           var_name='gtot', axes=axs[0, 1])

axs[1, 0].title.set_text('Vm on 1st layer')
plot_state(statemon_test_neurons1.t,
           statemon_test_neurons1.Vm[0],
           var_name='Vm', axes=axs[1, 0])

axs[1, 1].title.set_text('PSCs on 2nd layer')
plot_state(statemon_test_neurons2.t,
           statemon_test_neurons2.gtot[0],
           var_name='gtot', axes=axs[1, 1])

axs[2, 0].title.set_text('spikes on 2nd layer')
brian_plot(spikemon_test_neurons2, axes=axs[2, 0])

axs[2, 1].title.set_text('Vm on 2nd layer')
plot_state(statemon_test_neurons2.t,
           statemon_test_neurons2.Vm[0],
           var_name='Vm', axes=axs[2, 1])

fig.suptitle('full precision neuron')
plt.pause(0.001)

fig, axs = plt.subplots(3, 2, sharex=True)

axs[0, 0].title.set_text('input spikes')
brian_plot(spikemon_input, axes=axs[0, 0])

axs[0, 1].title.set_text('PSCs on 1st layer')
plot_state(statemon_layer1_int8.t,
           statemon_layer1_int8.gtot[0],
           var_name='gtot', axes=axs[0, 1])
plot_state(statemon_layer1_int8.t,
           statemon_layer1_int8.gtot[1],
           var_name='gtot', axes=axs[0, 1])

axs[1, 0].title.set_text('Vm on 1st layer')
plot_state(statemon_layer1_int8.t,
           statemon_layer1_int8.Vm[0],
           var_name='Vm', axes=axs[1, 0])

axs[1, 1].title.set_text('PSCs on 2nd layer')
plot_state(statemon_layer2_int8.t,
           statemon_layer2_int8.gtot[0],
           var_name='gtot', axes=axs[1, 1])

axs[2, 0].title.set_text('spikes on 2nd layer')
brian_plot(spikemon_layer2_int8, axes=axs[2, 0])

axs[2, 1].title.set_text('Vm on 2nd layer')
plot_state(statemon_layer2_int8.t,
           statemon_layer2_int8.Vm[0],
           var_name='Vm', axes=axs[2, 1])

fig.suptitle('8-bit fixed point neuron')
plt.pause(0.001)

fig, axs = plt.subplots(3, 2, sharex=True)

axs[0, 0].title.set_text('input spikes')
brian_plot(spikemon_input, axes=axs[0, 0])

axs[0, 1].title.set_text('PSCs on 1st layer')
plot_state(statemon_layer1_int4.t,
           statemon_layer1_int4.gtot[0],
           var_name='gtot', axes=axs[0, 1])
plot_state(statemon_layer1_int4.t,
           statemon_layer1_int4.gtot[1],
           var_name='gtot', axes=axs[0, 1])

axs[1, 0].title.set_text('Vm on 1st layer')
plot_state(statemon_layer1_int4.t,
           statemon_layer1_int4.Vm[0],
           var_name='Vm', axes=axs[1, 0])

axs[1, 1].title.set_text('PSCs on 2nd layer')
plot_state(statemon_layer2_int4.t,
           statemon_layer2_int4.gtot[0],
           var_name='gtot', axes=axs[1, 1])

axs[2, 0].title.set_text('spikes on 2nd layer')
brian_plot(spikemon_layer2_int4, axes=axs[2, 0])

axs[2, 1].title.set_text('Vm on 2nd layer')
plot_state(statemon_layer2_int4.t,
           statemon_layer2_int4.Vm[0],
           var_name='Vm', axes=axs[2, 1])

fig.suptitle('4-bit fixed-point precision neuron')
plt.pause(0.001)

""" profiling shows that FP8 model is around 20 times slower """
profiling_summary(show=15)
