""" This file has modified versions of some code from Teili. See
M. Milde, A. Renner, R. Krause, A. M. Whatley, S. Solinas, D. Zendrikov,
N. Risi, M. Rasetto, K. Burelo, V. R. C. Leite. teili: A toolbox for building
and testing neural algorithms and computational primitives using spiking neurons.
Unreleased software, Institute of Neuroinformatics, University of Zurich and ETH
Zurich, 2018.
"""

import numpy as np

from brian2 import run, device

from core.utils.misc import minifloat2decimal

from core.equations.neurons.LIF import LIF
from core.equations.synapses.CUBA import CUBA
from core.equations.synapses.STDP import STDP
from core.equations.neurons.fp8LIF import fp8LIF
from core.equations.synapses.fp8CUBA import fp8CUBA
from core.equations.synapses.fp8STDP import fp8STDP
from core.builder.groups_builder import create_synapses, create_neurons

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
    t_pre_homoeotasis_1 = np.arange(3, 304, isi)
    t_pre_weakLTP = np.arange(403, 604, isi)
    t_pre_weakLTD = np.arange(703, 904, isi)
    t_pre_strongLTP = np.arange(1003, 1204, isi)
    t_pre_strongLTD = np.arange(1303, 1504, isi)
    t_pre_homoeotasis_2 = np.arange(1603, 1804, isi)
    t_pre = np.hstack((t_pre_homoeotasis_1, t_pre_weakLTP, t_pre_weakLTD,
                       t_pre_strongLTP, t_pre_strongLTD, t_pre_homoeotasis_2))

    # Normal distributed shift of spike times to ensure homoeotasis
    t_post_homoeotasis_1 = t_pre_homoeotasis_1 + \
        np.random.randint(-3, 3, len(t_pre_homoeotasis_1))
    t_post_weakLTP = t_pre_weakLTP + 5   # post neuron spikes 7 ms after pre
    t_post_weakLTD = t_pre_weakLTD - 5   # post neuron spikes 7 ms before pre
    t_post_strongLTP = t_pre_strongLTP + 1  # post neurons spikes 1 ms after pre
    t_post_strongLTD = t_pre_strongLTD - 1  # post neurons spikes 1 ms before pre
    t_post_homoeotasis_2 = t_pre_homoeotasis_2 + \
        np.random.randint(-3, 3, len(t_pre_homoeotasis_2))

    t_post = np.hstack((t_post_homoeotasis_1, t_post_weakLTP, t_post_weakLTD,
                        t_post_strongLTP, t_post_strongLTD, t_post_homoeotasis_2))
    ind_pre = np.zeros(len(t_pre))
    ind_post = np.zeros(len(t_post))

    pre = SpikeGeneratorGroup(
        1, indices=ind_pre, times=t_pre * ms, name='gPre')
    post = SpikeGeneratorGroup(
        1, indices=ind_post, times=t_post * ms, name='gPost')
    return pre, post

def stdp(args):
    defaultclock.dt = args.timestep * ms

    pre_spikegenerator, post_spikegenerator = stimuli(isi=30)

    neuron_model = fp8LIF()
    # TODO 3ms refrac?
    pre_neurons = create_neurons(2, neuron_model)
    post_neurons = create_neurons(2, neuron_model)

    synapse_model = fp8CUBA()
    # TODO strong input and 3ms tau?
    pre_synapse = create_synapses(pre_spikegenerator, pre_neurons, synapse_model)
    post_synapse = create_synapses(post_spikegenerator, post_neurons, synapse_model)

    stdp_model = fp8STDP()

    run...
    if args.backend == 'cpp_standalone':
        device.build(args.code_path)




old_pattern = 'gtot0_post = g'
new_pattern = 'gtot1_post = g'
stdp_model.modify_model('model', new_pattern, old_pattern)
stdp_model.connection['condition'] = "i==j"
stdp_model.namespace['tau_itrace'] = 3 * ms
stdp_model.namespace['tau_jtrace'] = 3 * ms
stdp_model.parameters['tau_syn'] = 3*ms
stdp_synapse = create_synapses(pre_neurons, post_neurons, stdp_model)

# Setting up monitors
spikemon_pre_neurons = SpikeMonitor(pre_neurons, name='spikemon_pre_neurons')
statemon_pre_neurons = StateMonitor(pre_neurons, variables='Vm',
                                    record=0, name='statemon_pre_neurons')

spikemon_post_neurons = SpikeMonitor(
    post_neurons, name='spikemon_post_neurons')
statemon_post_neurons = StateMonitor(
    post_neurons, variables='Vm', record=0, name='statemon_post_neurons')


statemon_pre_synapse = StateMonitor(
    pre_synapse, variables=['g'], record=0, name='statemon_pre_synapse')

statemon_post_synapse = StateMonitor(stdp_synapse, variables=[
    'g', 'w_plast', 'i_trace', 'j_trace'],
    record=[0, 1], name='statemon_post_synapse')

duration = 2.
run(duration * second)


# Visualize
win_stdp = pg.GraphicsWindow(title="STDP Unit Test")
win_stdp.resize(2500, 1500)
win_stdp.setWindowTitle("Spike Time Dependent Plasticity")

p1 = win_stdp.addPlot()
win_stdp.nextRow()
p2 = win_stdp.addPlot()
win_stdp.nextRow()
p3 = win_stdp.addPlot()

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
            QtApp=QtApp,
            mainfig=win_stdp,
            subfig_rasterplot=p1)

Lineplot(DataModel_to_x_and_y_attr=[(statemon_post_synapse, ('t', 'w_plast'))],
            MyPlotSettings=PlotSettings(colors=['g']),
            x_range=(0, duration),
            title="Plastic synaptic weight",
            xlabel="Time (s)",
            ylabel="Synpatic weight w_plast",
            backend='pyqtgraph',
            QtApp=QtApp,
            mainfig=win_stdp,
            subfig=p2)

datamodel = StateVariablesModel(state_variable_names=['g'],
                                state_variables=[np.asarray(statemon_post_synapse.g[1])],
                                state_variables_times=[np.asarray(statemon_post_synapse.t)])
Lineplot(DataModel_to_x_and_y_attr=[(datamodel, ('t_g', 'g'))],
            MyPlotSettings=PlotSettings(colors=['m']),
            x_range=(0, duration),
            title="Post synaptic current",
            xlabel="Time (s)",
            ylabel="Synapic current I (pA)",
            backend='pyqtgraph',
            QtApp=QtApp,
            mainfig=win_stdp,
            subfig=p3,
            show_immediately=True)
