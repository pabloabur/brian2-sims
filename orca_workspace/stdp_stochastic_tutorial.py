# -*- coding: utf-8 -*-
# @Author: mmilde
# @Date:   2018-01-16 17:57:35

"""
This file provides an example of how to use neuron and synapse models which are present
on neurmorphic chips in the context of synaptic plasticity based on precise timing of spikes.
We use a standard STDP protocal with a exponentioally decaying window.

"""
import pyqtgraph as pg
import numpy as np

from brian2 import ms, us, second, pA, prefs,\
    SpikeMonitor, SpikeGeneratorGroup, StateMonitor, defaultclock, mA, ExplicitStateUpdater

from teili.core.groups import Neurons, Connections
from teili import TeiliNetwork
from teili.models.neuron_models import QuantStochLIF as LIF
from teili.models.synapse_models import QuantStochSynStdp as STDP
from teili.models.synapse_models import QuantStochSyn as SYN
from teili.stimuli.testbench import STDP_Testbench

from teili.tools.visualizer.DataViewers import PlotSettings
from teili.tools.visualizer.DataModels import StateVariablesModel
from teili.tools.visualizer.DataControllers import Lineplot, Rasterplot

def stimuli(isi=10):
    """Stimulus gneration for STDP protocols.

    This function returns two brian2 objects.
    Both are Spikegeneratorgroups which hold a single index each
    and varying spike times.
    The protocol follows homoeostasis, weak LTP, weak LTD, strong LTP,
    strong LTD, homoeostasis.

    Args:
        isi (int, optional): Interspike Interval. How many spikes per stimulus phase.

    Returns:
        SpikeGeneratorGroup (brian2.obj: Brian2 objects which hold the spiketimes and
            the respective neuron indices.
    """
    t_pre_homoeotasis_1 = np.arange(1, 302, isi)
    t_pre_weakLTP = np.arange(401, 602, isi)
    t_pre_weakLTD = np.arange(701, 902, isi)
    t_pre_strongLTP = np.arange(1001, 1202, isi)
    t_pre_strongLTD = np.arange(1301, 1502, isi)
    t_pre_homoeotasis_2 = np.arange(1601, 1802, isi)
    t_pre = np.hstack((t_pre_homoeotasis_1, t_pre_weakLTP, t_pre_weakLTD,
                       t_pre_strongLTP, t_pre_strongLTD, t_pre_homoeotasis_2))

    # Normal distributed shift of spike times to ensure homoeotasis
    t_post_homoeotasis_1 = t_pre_homoeotasis_1 + \
        np.clip(np.random.randn(len(t_pre_homoeotasis_1)), -1, 1)
    t_post_weakLTP = t_pre_weakLTP + 5   # post neuron spikes 7 ms after pre
    t_post_weakLTD = t_pre_weakLTD - 5   # post neuron spikes 7 ms before pre
    t_post_strongLTP = t_pre_strongLTP + 1  # post neurons spikes 1 ms after pre
    t_post_strongLTD = t_pre_strongLTD - 1  # post neurons spikes 1 ms before pre
    t_post_homoeotasis_2 = t_pre_homoeotasis_2 + \
        np.clip(np.random.randn(len(t_pre_homoeotasis_2)), -1, 1)

    t_post = np.hstack((t_post_homoeotasis_1, t_post_weakLTP, t_post_weakLTD,
                        t_post_strongLTP, t_post_strongLTD, t_post_homoeotasis_2))
    ind_pre = np.zeros(len(t_pre))
    ind_post = np.zeros(len(t_post))

    pre = SpikeGeneratorGroup(
        1, indices=ind_pre, times=t_pre * ms, name='gPre')
    post = SpikeGeneratorGroup(
        1, indices=ind_post, times=t_post * ms, name='gPost')
    return pre, post

prefs.codegen.target = "numpy"
defaultclock.dt = 1*ms
Net = TeiliNetwork()

pre_spikegenerator, post_spikegenerator = stimuli(isi=15)

pre_neurons = Neurons(2, equation_builder=LIF(num_inputs=1),
                      method=ExplicitStateUpdater('''x_new = f(x,t)'''),
                      name='pre_neurons')

post_neurons = Neurons(2, equation_builder=LIF(num_inputs=2),
                       method=ExplicitStateUpdater('''x_new = f(x,t)'''),
                       name='post_neurons')


pre_synapse = Connections(pre_spikegenerator, pre_neurons,
                          method=ExplicitStateUpdater('''x_new = f(x,t)'''),
                          equation_builder=SYN(), name='pre_synapse')

post_synapse = Connections(post_spikegenerator, post_neurons,
                           method=ExplicitStateUpdater('''x_new = f(x,t)'''),
                           equation_builder=SYN(), name='post_synapse')

stdp_synapse = Connections(pre_neurons, post_neurons,
                           method=ExplicitStateUpdater('''x_new = f(x,t)'''),
                           equation_builder=STDP, name='stdp_synapse')

pre_synapse.connect(True)
post_synapse.connect(True)
# Set parameters:
stdp_synapse.connect("i==j")
stdp_synapse.w_plast = 1 # Only necessary in this testbench
stdp_synapse.taupre = 5*ms
stdp_synapse.taupost = 5*ms
stdp_synapse.stdp_thres = 21
stdp_synapse.rand_num_bits_Apre = 4
stdp_synapse.rand_num_bits_Apost = 4
stdp_synapse.tausyn = 3*ms
pre_synapse.gain_syn = 70*mA
pre_synapse.tausyn = 3*ms
post_synapse.gain_syn = 70*mA
post_synapse.tausyn = 3*ms
pre_neurons.tau = 3*ms
post_neurons.tau = 3*ms

# Setting up monitors
spikemon_pre_neurons = SpikeMonitor(pre_neurons, name='spikemon_pre_neurons')

spikemon_post_neurons = SpikeMonitor(
    post_neurons, name='spikemon_post_neurons')


statemon_pre_synapse = StateMonitor(
    pre_synapse, variables=['I_syn'], record=0, name='statemon_pre_synapse')

statemon_post_synapse = StateMonitor(stdp_synapse, variables=[
    'I_syn', 'w_plast', 'weight'],
    record=True, name='statemon_post_synapse')

Net.add(pre_spikegenerator, post_spikegenerator,
        pre_neurons, post_neurons,
        pre_synapse, post_synapse, stdp_synapse,
        spikemon_pre_neurons, spikemon_post_neurons,
        statemon_pre_synapse, statemon_post_synapse)

duration = 2.
Net.run(duration * second)


# Visualize
win_stdp = pg.GraphicsWindow(title="STDP Unit Test")
win_stdp.resize(2500, 1500)
win_stdp.setWindowTitle("Spike Time Dependent Plasticity")

p1 = win_stdp.addPlot()
win_stdp.nextRow()
p2 = win_stdp.addPlot()
p2.setXLink(p1)
win_stdp.nextRow()
p3 = win_stdp.addPlot()
p3.setXLink(p2)

text1 = pg.TextItem(text='Homoeostasis', anchor=(-0.3, 0.5))
text2 = pg.TextItem(text='Weak Pot.', anchor=(-0.3, 0.5))
text3 = pg.TextItem(text='Weak Dep.', anchor=(-0.3, 0.5))
text4 = pg.TextItem(text='Strong Pot.', anchor=(-0.3, 0.5))
text5 = pg.TextItem(text='Strong Dep.', anchor=(-0.3, 0.5))
text6 = pg.TextItem(text='Homoeostasis', anchor=(-0.3, 0.5))
p1.addItem(text1)
p1.addItem(text2)
p1.addItem(text3)
p1.addItem(text4)
p1.addItem(text5)
p1.addItem(text6)
text1.setPos(0, 0.5)
text2.setPos(0.300, 0.5)
text3.setPos(0.600, 0.5)
text4.setPos(0.900, 0.5)
text5.setPos(1.200, 0.5)
text6.setPos(1.500, 0.5)

Rasterplot(MyEventsModels=[spikemon_pre_neurons, spikemon_post_neurons],
            MyPlotSettings=PlotSettings(colors=['w', 'r']),
            time_range=(0, duration),
            neuron_id_range=(-1, 2),
            title="STDP protocol",
            xlabel="Time (s)",
            ylabel="Neuron ID",
            backend='pyqtgraph',
            mainfig=win_stdp,
            subfig_rasterplot=p1)

Lineplot(DataModel_to_x_and_y_attr=[(statemon_post_synapse, ('t', 'w_plast'))],
            MyPlotSettings=PlotSettings(colors=['g']),
            x_range=(0, duration),
            title="Plastic synaptic weight",
            xlabel="Time (s)",
            ylabel="Synpatic weight w_plast",
            backend='pyqtgraph',
            mainfig=win_stdp,
            subfig=p2)

datamodel = StateVariablesModel(state_variable_names=['I_syn'],
                                state_variables=[np.asarray(statemon_post_synapse.I_syn[1])],
                                state_variables_times=[np.asarray(statemon_post_synapse.t)])
Lineplot(DataModel_to_x_and_y_attr=[(datamodel, ('t_I_syn', 'I_syn'))],
            MyPlotSettings=PlotSettings(colors=['m']),
            x_range=(0, duration),
            title="Post synaptic current",
            xlabel="Time (s)",
            ylabel="Synapic current I (pA)",
            backend='pyqtgraph',
            mainfig=win_stdp,
            subfig=p3,
            show_immediately=True)
