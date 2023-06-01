""" This file has modified versions of some code from Teili. See
M. Milde, A. Renner, R. Krause, A. M. Whatley, S. Solinas, D. Zendrikov,
N. Risi, M. Rasetto, K. Burelo, V. R. C. Leite. teili: A toolbox for building
and testing neural algorithms and computational primitives using spiking
neurons. Unreleased software, Institute of Neuroinformatics, University of
Zurich and ETH Zurich, 2018.
"""

import numpy as np
import feather
import pandas as pd

from brian2 import run, device, second, Hz, ms, defaultclock,\
    SpikeGeneratorGroup, StateMonitor, SpikeMonitor, TimedArray, PoissonGroup

import plotext as plt

from core.utils.misc import minifloat2decimal, decimal2minifloat

from core.equations.neurons.fp8LIF import fp8LIF
from core.equations.synapses.fp8CUBA import fp8CUBA
from core.equations.synapses.fp8STDP import fp8STDP
from core.builder.groups_builder import create_synapses, create_neurons

from brian2 import Function, DEFAULT_FUNCTIONS
minifloat2decimal = Function(minifloat2decimal, arg_units=[1], return_unit=1)
DEFAULT_FUNCTIONS.update({'minifloat2decimal': minifloat2decimal})


def stimuli_protocol1(isi=10):
    """Stimulus gneration for STDP protocols.

    This function returns two brian2 objects.
    Both are Spikegeneratorgroups which hold a single index each
    and varying spike times.
    The protocol follows homoeostasis, weak LTP, weak LTD, strong LTP,
    strong LTD, homoeostasis.

    Args:
        isi (int, optional): Interspike Interval. How many spikes per stimulus
            phase.

    Returns:
        SpikeGeneratorGroup (brian2.obj: Brian2 objects which hold the
            spiketimes and the respective neuron indices.
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
    t_post_strongLTP = t_pre_strongLTP + 1  # post neurons spike 1 ms after pre
    t_post_strongLTD = t_pre_strongLTD - 1  # and 1 ms before pre
    t_post_homoeotasis_2 = t_pre_homoeotasis_2 + \
        np.random.randint(-3, 3, len(t_pre_homoeotasis_2))

    t_post = np.hstack((t_post_homoeotasis_1, t_post_weakLTP, t_post_weakLTD,
                        t_post_strongLTP, t_post_strongLTD,
                        t_post_homoeotasis_2))
    ind_pre = np.zeros(len(t_pre))
    ind_post = np.zeros(len(t_post))

    pre = SpikeGeneratorGroup(
        1, indices=ind_pre, times=t_pre * ms, name='gPre')
    post = SpikeGeneratorGroup(
        1, indices=ind_post, times=t_post * ms, name='gPost')

    return pre, post


def stimuli_protocol2():
    trials = 105
    trial_duration = 60
    N = trial_duration
    wait_time = 2*trial_duration  # delay to avoid interferences
    tmax = trial_duration*trials + wait_time*trials

    # Define matched spike times between pre and post neurons
    post_tspikes = np.arange(1, N*trials + 1).reshape((trials, N))
    pre_tspikes = post_tspikes[:, np.array(range(N-1, -1, -1))]
    # Use the ones below to test simultaneous samples from random function
    # post_tspikes = np.arange(0, trials, 2).reshape(-1, 1) + np.ones(N)
    # pre_tspikes = np.arange(1, trials, 2).reshape(-1, 1) + np.ones(N)

    # Create inputs arrays, which will be 1 when neurons are supposed to spike
    pre_input = np.zeros((tmax, N))
    post_input = np.zeros((tmax, N))
    for ind, spks in enumerate(pre_tspikes.T):
        for j, spk in enumerate(spks.astype(int)):
            pre_input[spk-1 + j*wait_time, ind] = 1
    for ind, spks in enumerate(post_tspikes.T):
        for j, spk in enumerate(spks.astype(int)):
            post_input[spk-1 + j*wait_time, ind] = 1

    tapre = TimedArray(pre_input, dt=defaultclock.dt)
    tapost = TimedArray(post_input, dt=defaultclock.dt)
    return tapre, tapost, tmax, N, trial_duration


def stdp(args):
    defaultclock.dt = args.timestep * ms
    run_namespace = {}

    if args.protocol == 1:
        N_pre, N_post = 2, 2
        n_conns = N_pre
        pre_spikegenerator, post_spikegenerator = stimuli_protocol1(isi=30)
        neuron_model = fp8LIF()
        pre_neurons = create_neurons(2, neuron_model)
        post_neurons = create_neurons(2, neuron_model)

        synapse_model = fp8CUBA()
        synapse_model.modify_model('parameters',
                                   decimal2minifloat(192),
                                   key='weight')
        pre_synapse = create_synapses(pre_spikegenerator,
                                      pre_neurons,
                                      synapse_model)
        post_synapse = create_synapses(post_spikegenerator,
                                       post_neurons,
                                       synapse_model)

        tmax = 2. * second

    elif args.protocol == 2:
        tapre, tapost, tmax, N, trial_duration = stimuli_protocol2()
        N_pre, N_post = N, N
        n_conns = N
        neuron_model = fp8LIF()
        neuron_model.modify_model(
            'model',
            'summed_decay = tapre(t, i)',
            old_expr='summed_decay = fp8_add(decay_term, gtot*int(not_refractory))')
        neuron_model.modify_model('threshold', '1', old_expr='Vthr')
        pre_neurons = create_neurons(N, neuron_model)

        neuron_model = fp8LIF()
        neuron_model.modify_model(
            'model',
            'summed_decay = tapost(t, i)',
            old_expr='summed_decay = fp8_add(decay_term, gtot*int(not_refractory))')
        neuron_model.modify_model('threshold', '1', old_expr='Vthr')
        post_neurons = create_neurons(N, neuron_model)

        tmax = tmax * ms

        run_namespace.update({'tapre': tapre, 'tapost': tapost})

    elif args.protocol == 3:
        N_pre = 1000
        N_post = 1
        n_conns = N_pre
        tmax = 10000 * ms
        pre_neurons = PoissonGroup(N_pre, 15*Hz)
        neuron_model = fp8LIF()
        post_neurons = create_neurons(N_post, neuron_model)

    if args.protocol == 1 or args.protocol == 2:
        conn_condition = 'i==j'
    if args.protocol == 3:
        # None makes it all to all
        conn_condition = None

    stdp_model = fp8STDP()
    stdp_model.modify_model('connection',
                            conn_condition,
                            key='condition')
    rng = np.random.default_rng()
    sampled_var = rng.uniform(0, 16, n_conns)

    if args.precision == 'fp64':
        # Turns plasticity mechanisms into full-precision
        stdp_model.modify_model('model', 'w_plast : 1', old_expr='w_plast : integer')
        stdp_model.modify_model('namespace', 0.001953125, key='eta')

        eq_delete = ['w_change = fp8_multiply(eta, Ca_post)',
                     'w_change = fp8_multiply(184, w_change)',
                     'w_change = fp8_add(w_plast, w_change)']
        for eq in eq_delete:
            stdp_model.modify_model('on_pre', '', old_expr=eq)
        stdp_model.modify_model(
            'on_pre',
            'w_plast = clip(w_plast - eta*minifloat2decimal(Ca_post), 0, inf)\n',
            old_expr='w_plast = int(w_change<=127)*w_change')

        stdp_model.modify_model('on_post',
                                '',
                                old_expr='w_change = fp8_multiply(eta, Ca_pre)')
        stdp_model.modify_model(
            'on_post',
            'w_plast = clip(w_plast + eta*minifloat2decimal(Ca_pre), 0, inf)\n',
            old_expr='w_plast = fp8_add(w_plast, w_change)')
    if args.precision == 'fp8':
        sampled_var = decimal2minifloat(sampled_var, raise_warning=False)

    stdp_model.modify_model('parameters',
                            sampled_var,
                            key='w_plast')

    if args.protocol == 3:
        stdp_model.model += ('dCa_syn/dt = fp8_multiply(Ca_syn, 55)/second : 1\n'
                             + 'spiked : second\n')
        stdp_model.on_pre += ('Ca_syn = fp8_add(Ca_syn, 127)\n'
                              + 'spiked = t\n')
        stdp_model.modify_model('on_post', 'Ca_syn', old_expr='Ca_pre')

    stdp_synapse = create_synapses(pre_neurons, post_neurons, stdp_model)

    # Setting up monitors
    spikemon_pre_neurons = SpikeMonitor(pre_neurons,
                                        name='spikemon_pre_neurons')
    spikemon_post_neurons = SpikeMonitor(post_neurons,
                                         name='spikemon_post_neurons')
    if args.protocol != 3:
        statemon_pre_neurons = StateMonitor(pre_neurons,
                                            variables=['Vm', 'g', 'Ca'],
                                            record=range(N_pre),
                                            name='statemon_pre_neurons')
    statemon_post_neurons = StateMonitor(post_neurons,
                                         variables=['Vm', 'g', 'Ca'],
                                         record=range(N_post),
                                         name='statemon_post_neurons')
    statemon_post_synapse = StateMonitor(stdp_synapse,
                                         variables=['w_plast'],
                                         record=range(N_pre),
                                         name='statemon_post_synapse')

    run(tmax, namespace=run_namespace)

    if args.backend == 'cpp_standalone':
        device.build(args.code_path)

    if not args.quiet:
        if args.protocol == 1:
            plt.scatter(spikemon_pre_neurons.t/ms, spikemon_pre_neurons.i)
            plt.scatter(spikemon_post_neurons.t/ms, spikemon_post_neurons.i)
            plt.show()

            plt.clear_figure()
            plt.plot(statemon_post_synapse.t/ms,
                     minifloat2decimal(statemon_post_synapse.w_plast[0]))
            plt.show()

            plt.clear_figure()
            plt.plot(statemon_post_neurons.t/ms,
                     minifloat2decimal(statemon_post_neurons.Ca[0]))
            plt.show()

            plt.clear_figure()
            plt.plot(statemon_pre_neurons.t/ms,
                     minifloat2decimal(statemon_pre_neurons.Vm[0]))
            plt.show()

        elif args.protocol == 2:
            labels = ('indices', 'times', 'type')
            data = [(np.array(spikemon_pre_neurons.i),
                     spikemon_pre_neurons.t/ms,
                     ['pre' for _ in range(len(spikemon_pre_neurons.t))]),
                    (np.array(spikemon_post_neurons.i),
                     spikemon_post_neurons.t/ms,
                     ['post' for _ in range(len(spikemon_post_neurons.t))])
                    ]
            temp_data = {lbl: [] for lbl in labels}
            for dat in data:
                for idx in range(len(dat)):
                    temp_data[labels[idx]].extend(dat[idx])
            spikes = pd.DataFrame(temp_data)
            feather.write_dataframe(spikes, args.save_path + 'spikes.feather')

            pairs_timing = (spikemon_post_neurons.t[:trial_duration]
                            - spikemon_post_neurons.t[:trial_duration][::-1])/ms
            plt.plot(pairs_timing,
                     minifloat2decimal(statemon_post_synapse.w_plast[:, -1]))
            plt.show()

            plt.clear_figure()
            plt.plot(minifloat2decimal(statemon_pre_neurons.Ca[0][:100]))
            plt.plot(minifloat2decimal(statemon_post_neurons.Ca[0][:100]))
            plt.show()

            plt.clear_figure()
            plt.plot(minifloat2decimal(statemon_pre_neurons.Ca[29][:100]))
            plt.plot(minifloat2decimal(statemon_post_neurons.Ca[29][:100]))
            plt.show()

        elif args.protocol == 3:
            plt.hist(statemon_post_synapse.w_plast[:, -1])
            plt.show()
            plt.clear_figure()
            plt.plot(minifloat2decimal(statemon_post_synapse.w_plast[0, :]))
            plt.show()
