""" This file has modified versions of some code from Teili. See
M. Milde, A. Renner, R. Krause, A. M. Whatley, S. Solinas, D. Zendrikov,
N. Risi, M. Rasetto, K. Burelo, V. R. C. Leite. teili: A toolbox for building
and testing neural algorithms and computational primitives using spiking neurons.
Unreleased software, Institute of Neuroinformatics, University of Zurich and ETH
Zurich, 2018.
"""

import numpy as np

from brian2 import run, device, second, ms, defaultclock, SpikeGeneratorGroup,\
    StateMonitor, SpikeMonitor

from brian2tools import brian_plot, plot_state
import matplotlib.pyplot as plt

from core.utils.misc import minifloat2decimal, decimal2minifloat

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
    # TODO 3ms tau?
    synapse_model.modify_model('parameters',
                               decimal2minifloat(192),
                               key='weight')
    pre_synapse = create_synapses(pre_spikegenerator, pre_neurons, synapse_model)
    post_synapse = create_synapses(post_spikegenerator, post_neurons, synapse_model)

    stdp_model = fp8STDP()
    stdp_model.modify_model('connection', "i==j", key='condition')
    stdp_model.modify_model('parameters',
                            decimal2minifloat(0.001953125),
                            key='w_plast')
    # TODO tau syn and stdp tau 3ms?
    stdp_synapse = create_synapses(pre_neurons, post_neurons, stdp_model)

    # Setting up monitors
    spikemon_pre_neurons = SpikeMonitor(pre_neurons,
                                        name='spikemon_pre_neurons')
    spikemon_post_neurons = SpikeMonitor(post_neurons,
                                         name='spikemon_post_neurons')
    statemon_pre_neurons = StateMonitor(pre_neurons,
                                        variables=['Vm', 'g', 'Ca'],
                                        record=0,
                                        name='statemon_pre_neurons')
    statemon_post_neurons = StateMonitor(post_neurons,
                                         variables=['Vm', 'g', 'Ca'],
                                         record=0,
                                         name='statemon_post_neurons')
    statemon_post_synapse = StateMonitor(stdp_synapse,
                                         variables=['w_plast'],
                                         record=[0, 1],
                                         name='statemon_post_synapse')


    duration = 2.
    run(duration * second)
    if args.backend == 'cpp_standalone':
        device.build(args.code_path)

    if not args.quiet:
        # TODO why does w decays to zero after first update?
        # TODO why is dec2mf returning ...5 at the end of conversion??
        brian_plot(spikemon_pre_neurons)

        plt.figure()
        brian_plot(spikemon_post_neurons)

        plt.figure()
        plot_state(statemon_post_synapse.t,
                   minifloat2decimal(statemon_post_synapse.w_plast[0]),
                   var_name='weight')

        plt.figure()
        plot_state(statemon_post_neurons.t,
                   minifloat2decimal(statemon_post_neurons.Ca[0]),
                   var_name='Time window')

        plt.figure()
        plot_state(statemon_pre_neurons.t,
                   minifloat2decimal(statemon_pre_neurons.Vm[0]),
                   var_name='Vm')

        plt.show()
