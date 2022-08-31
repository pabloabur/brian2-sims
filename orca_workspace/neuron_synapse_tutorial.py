#!/usr/bin/env python3
from pyqtgraph.Qt import QtGui
import pyqtgraph as pg
import numpy as np
import sys
from brian2 import mV, uA, mA, ms, ohm, second, pA, nA, prefs,\
    SpikeMonitor, StateMonitor, SpikeGeneratorGroup, defaultclock,\
    ExplicitStateUpdater, run, set_device, profiling_summary

from brian2tools import plot_state, brian_plot
import matplotlib.pyplot as plt

from teili.tools.misc import decimal2minifloat, minifloat2decimal
from teili.tools.misc import DEFAULT_FUNCTIONS
from teili.tools.visualizer.DataModels import StateVariablesModel

from equations.neurons.LIF import LIF
from equations.synapses.CUBA import CUBA
from equations.neurons.fp8LIF import fp8LIF
from equations.synapses.fp8CUBA import fp8CUBA
from builder.groups_builder import create_synapses, create_neurons

from teili.tools.visualizer.DataViewers import PlotSettings
from teili.tools.visualizer.DataControllers import Rasterplot, Lineplot
from teili.tools.lfsr import create_lfsr

prefs.codegen.target = "numpy"
#set_device('cpp_standalone')
defaultclock.dt = 1*ms
method = ExplicitStateUpdater('''x_new = f(x,t)''')

input_timestamps = np.asarray([1, 3, 4, 5, 6, 7, 8, 9]) * ms
input_indices = np.asarray([0, 0, 0, 0, 0, 0, 0, 0])
input_spikegenerator = SpikeGeneratorGroup(1, indices=input_indices,
                                           times=input_timestamps, 
                                           name='input_spikegenerator')


# Preparing standard models
neuron_model = LIF()
test_neurons1 = create_neurons(2, neuron_model, name='std_neu1')
test_neurons2 = create_neurons(2, neuron_model, name='std_neu2')
test_neurons1.Iconst = 260*pA

synapse_model_test = CUBA()
input_synapse = create_synapses(input_spikegenerator, test_neurons1,
                                synapse_model_test, name='std_syn1')
input_synapse.weight = 80*mV
test_synapse = create_synapses(test_neurons1, test_neurons2,
                               synapse_model_test, name='std_syn2')
test_synapse.weight = 50*mV

# Preparing 8-bit models
neuron_model2 = fp8LIF()
test_neurons3 = create_neurons(2, neuron_model2, name='fp8_neu1')
test_neurons4 = create_neurons(2, neuron_model2, name='fp8_neu2')
test_neurons3.Iconst = decimal2minifloat(52)

synapse_model_test2 = fp8CUBA()
input_synapse2 = create_synapses(input_spikegenerator, test_neurons3,
                                 synapse_model_test2, name='fp8_syn1')
input_synapse2.weight = decimal2minifloat(80)
test_synapse2 = create_synapses(test_neurons3, test_neurons4,
                                synapse_model_test2, name='fp8_syn2')
test_synapse2.weight = decimal2minifloat(52)

# TODO fix this; it is just to keep old stochastic parameters
sim_type = 'not_stochastic' 
if 'stochastic_decay' in sim_type:
    # Example of how to set a single parameter
    # Fast neuron to allow more spikes
    test_neurons1.refrac_tau = 1 * ms
    test_neurons1.refrac_decay_numerator = 128
    test_neurons2.refrac_tau = 1 * ms
    test_neurons2.refrac_decay_numerator = 128
    test_neurons1.tau = 20 * ms
    test_neurons1.decay_numerator = 243
    test_neurons2.tau = 20 * ms
    test_neurons2.decay_numerator = 243
    # long EPSC or big weight to allow summations
    test_neurons1.tausyn = 5*ms
    test_neurons1.syn_decay_numerator = 213
    test_neurons2.tausyn = 10*ms
    test_neurons2.syn_decay_numerator = 233
    input_synapse.weight = 15
    test_synapse.weight = 7
    test_neurons1.Iconst = 11.0 * mA
    test_neurons1.Vm = 3*mV
    test_neurons2.Vm = 3*mV
    test_neurons1.g_psc = 2 * ohm
    test_neurons2.g_psc = 2 * ohm
    syn_variables = 'I'

if 'lfsr' in sim_type:
        num_bits = 4
        test_neurons1.rand_num_bits = num_bits
        test_neurons2.rand_num_bits = num_bits
if 'lfsr' in sim_type:
        num_bits = 4
        input_synapse.rand_num_bits_syn = num_bits
        test_synapse.rand_num_bits_syn = num_bits
        ta = create_lfsr([test_neurons1, test_neurons2], [input_synapse, test_synapse], defaultclock.dt)

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
statemon_input_synapse2 = StateMonitor(
    test_neurons3, variables='gtot', record=True)
spikemon_test_neurons4 = SpikeMonitor(test_neurons4)
statemon_test_synapse2 = StateMonitor(
    test_neurons4, variables='gtot', record=True)
statemon_test_neurons3 = StateMonitor(test_neurons3, variables=[
    'gtot', "Vm"], record=[0, 1])
statemon_test_neurons4 = StateMonitor(test_neurons4,
                                      variables=['Vm', 'gtot'],
                                      record=0)

duration = 0.5
run(duration * second, namespace={}, profile=True)

fig, axs = plt.subplots(3, 2, sharex=True)

axs[0, 0].title.set_text('input spikes')
brian_plot(spikemon_input, axes=axs[0, 0])

axs[0, 1].title.set_text('PSCs on 1st layer')
plot_state(statemon_test_neurons3.t,
           minifloat2decimal(statemon_test_neurons3.gtot[0]),
           var_name='gtot', axes=axs[0, 1])
plot_state(statemon_test_neurons3.t,
           minifloat2decimal(statemon_test_neurons3.gtot[1]),
           var_name='gtot', axes=axs[0, 1])

axs[1, 0].title.set_text('Vm on 1st layer')
plot_state(statemon_test_neurons3.t,
           minifloat2decimal(statemon_test_neurons3.Vm[0]),
           var_name='Vm', axes=axs[1, 0])

axs[1, 1].title.set_text('PSCs on 2nd layer')
plot_state(statemon_test_neurons4.t,
           minifloat2decimal(statemon_test_neurons4.gtot[0]),
           var_name='gtot', axes=axs[1, 1])

axs[2, 0].title.set_text('spikes on 2nd layer')
brian_plot(spikemon_test_neurons4, axes=axs[2, 0])

axs[2, 1].title.set_text('Vm on 2nd layer')
plot_state(statemon_test_neurons4.t,
           minifloat2decimal(statemon_test_neurons4.Vm[0]),
           var_name='Vm', axes=axs[2, 1])

fig.suptitle('8-bit floating point neuron')
plt.pause(0.001)

# Visualize simulation results
app = QtGui.QApplication.instance()
if app is None:
    app = QtGui.QApplication(sys.argv)
else:
    print('QApplication instance already exists: %s' % str(app))

pg.setConfigOptions(antialias=True)
labelStyle = {'color': '#FFF', 'font-size': 12}
MyPlotSettings = PlotSettings(fontsize_title=labelStyle['font-size'],
                              fontsize_legend=labelStyle['font-size'],
                              fontsize_axis_labels=10,
                              marker_size=7)

win = pg.GraphicsWindow()
win.resize(2100, 1200)
win.setWindowTitle('Simple Spiking Neural Network')

p1 = win.addPlot(title="Input spike generator")
p2 = win.addPlot(title="Input synapses")
win.nextRow()
p3 = win.addPlot(title='Intermediate test neurons 1')
p4 = win.addPlot(title="Test synapses")
win.nextRow()
p5 = win.addPlot(title="Rasterplot of output test neurons 2")
p6 = win.addPlot(title="Output test neurons 2")


# Spike generator
Rasterplot(MyEventsModels=[spikemon_input],
           MyPlotSettings=MyPlotSettings,
           time_range=[0, duration],
           neuron_id_range=None,
           title="Input spike generator",
           xlabel='Time (ms)',
           ylabel="Neuron ID",
           backend='pyqtgraph',
           mainfig=win,
           subfig_rasterplot=p1,
           QtApp=app,
           show_immediately=False)

# Input synapses
data = [(statemon_input_synapse, ('t', 'gtot'))]
Lineplot(DataModel_to_x_and_y_attr=data,
         MyPlotSettings=MyPlotSettings,
         x_range=[0, duration],
         title="Input synapses",
         xlabel="Time (ms)",
         ylabel="EPSC (A)",
         backend='pyqtgraph',
         mainfig=win,
         subfig=p2,
         QtApp=app,
         show_immediately=False)

# Intermediate neurons
MyData_intermed_neurons = [(statemon_test_neurons1, ('t', 'Vm'))]
Lineplot(DataModel_to_x_and_y_attr=MyData_intermed_neurons,
         MyPlotSettings=MyPlotSettings,
         x_range=[0, duration],
         title='Intermediate test neurons 1',
         xlabel="Time (ms)",
         ylabel='Vm',
         backend='pyqtgraph',
         mainfig=win,
         subfig=p3,
         QtApp=app,
         show_immediately=False)

# Output synapses
data = [(statemon_test_neurons2, ('t', 'gtot'))]
Lineplot(DataModel_to_x_and_y_attr=data,
         MyPlotSettings=MyPlotSettings,
         x_range=[0, duration],
         title="Test synapses",
         xlabel="Time (ms)",
         ylabel="EPSC (A)",
         backend='pyqtgraph',
         mainfig=win,
         subfig=p4,
         QtApp=app,
         show_immediately=False)


Rasterplot(MyEventsModels=[spikemon_test_neurons2],
           MyPlotSettings=MyPlotSettings,
           time_range=[0, duration],
           neuron_id_range=None,
           title="Rasterplot of output test neurons 2",
           xlabel='Time (ms)',
           ylabel="Neuron ID",
           backend='pyqtgraph',
           mainfig=win,
           subfig_rasterplot=p5,
           QtApp=app,
           show_immediately=False)

MyData_output = [(statemon_test_neurons2, ('t', 'Vm'))]
Lineplot(DataModel_to_x_and_y_attr=MyData_output,
         MyPlotSettings=MyPlotSettings,
         x_range=[0, duration],
         title="Output test neurons 2",
         xlabel="Time (ms)",
         ylabel="Vm",
         backend='pyqtgraph',
         mainfig=win,
         subfig=p6,
         QtApp=app,
         show_immediately=False)

app.exec()

""" profiling shows that FP8 model is around 20 times slower """
profiling_summary(show=15)
