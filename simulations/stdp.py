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

from brian2 import run, device, defaultclock, scheduling_summary,\
    SpikeGeneratorGroup, StateMonitor, SpikeMonitor, TimedArray, PoissonGroup
from brian2 import mV, second, Hz, ms

import plotext as plt

from core.utils.misc import minifloat2decimal, decimal2minifloat

from core.equations.neurons.fp8LIF import fp8LIF
from core.equations.synapses.fp8CUBA import fp8CUBA
from core.equations.synapses.fp8STDP import fp8STDP
from core.equations.neurons.LIF import LIF
from core.equations.synapses.CUBA import CUBA
from core.equations.synapses.STDP import STDP
from core.equations.neurons.tsvLIF import tsvLIF
from core.equations.synapses.tsvCUBA import tsvCUBA
from core.equations.synapses.tsvSTDP import tsvSTDP
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

    # TODO not sure about this either
    static_weight = 80

    if args.precision == 'fp8':
        neuron_model = fp8LIF()
        aux_weight_sample = lambda x: decimal2minifloat(x, raise_warning=False)
        aux_plot = lambda x: minifloat2decimal(x)
        aux_plot_Ca = lambda x: minifloat2decimal(x)
        synapse_model = fp8CUBA()
        stdp_model = fp8STDP()
    if args.precision == 'fp64':
        neuron_model = tsvLIF()
        aux_weight_sample = lambda x: x * mV
        aux_plot = lambda x: x/mV
        aux_plot_Ca = lambda x: x
        synapse_model = tsvCUBA()
        stdp_model = tsvSTDP()

    if args.protocol == 1:
        N_pre, N_post = 2, 2
        n_conns = N_pre
        pre_spikegenerator, post_spikegenerator = stimuli_protocol1(isi=30)
        pre_neurons = create_neurons(2, neuron_model)
        post_neurons = create_neurons(2, neuron_model)

        neuron_model = LIF()
        ref_pre_neurons = create_neurons(2, neuron_model)
        ref_post_neurons = create_neurons(2, neuron_model)

        synapse_model.modify_model('parameters',
                                   aux_weight_sample(static_weight),
                                   key='weight')
        pre_synapse = create_synapses(pre_spikegenerator,
                                      pre_neurons,
                                      synapse_model,
                                      name='static_pre_synapse')
        post_synapse = create_synapses(post_spikegenerator,
                                       post_neurons,
                                       synapse_model,
                                       name='static_post_synapse')

        synapse_model = CUBA()
        synapse_model.modify_model('parameters',
                                   static_weight*mV,
                                   key='weight')
        ref_pre_synapse = create_synapses(pre_spikegenerator,
                                          ref_pre_neurons,
                                          synapse_model,
                                          name='ref_static_pre_synapse')
        ref_post_synapse = create_synapses(post_spikegenerator,
                                           ref_post_neurons,
                                           synapse_model,
                                           name='ref_static_post_synapse')

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

    stdp_model.modify_model('connection',
                            conn_condition,
                            key='condition')

    rng = np.random.default_rng()
    sampled_weights = rng.uniform(0, 16, n_conns)
    if args.precision == 'fp8':
        sampled_weights = aux_weight_sample(sampled_weights)
    if args.precision == 'fp64':
        sampled_weights = aux_weight_sample(sampled_weights)
    stdp_model.modify_model('parameters',
                            sampled_weights,
                            key='w_plast')

    if args.protocol == 3:
        stdp_model.model += ('dCa_syn/dt = fp8_multiply(Ca_syn, 55)/second : 1\n'
                             + 'spiked : second\n')
        stdp_model.on_pre += ('Ca_syn = fp8_add(Ca_syn, 127)\n'
                              + 'spiked = t\n')
        stdp_model.modify_model('on_post', 'Ca_syn', old_expr='Ca_pre')

    # TODO should also find right ca_inc
    pre_neurons.namespace['Ca_inc'] = 1
    post_neurons.namespace['Ca_inc'] = 1
    stdp_synapse = create_synapses(pre_neurons,
                                   post_neurons,
                                   stdp_model,
                                   name='stdp_synapse')

    # TODO organize fp8 as well
    # TODO this should be a module/function,  e.g. add_tsv_scheme()
    # TODO not sure this is the eta we want
    stdp_synapse.namespace['eta'] = 0.01*mV
    if args.precision == 'fp64':
        stdp_synapse.run_regularly(
            '''delta_w = int(Ca_pre<0 and Ca_post>0)*(w_plast - eta*Ca_post) + int(Ca_pre>0 and Ca_post<0)*(w_plast + eta*Ca_post)
               w_plast += delta_w
               ''',
            name='weight_update',
            dt=defaultclock.dt,
            when='after_resets',
            order=0)
        #pre_neurons.run_regularly(
        #    'Ca = Ca*int(Ca>0) - Ca*int(Ca<0)',
        #    name='clear_spike_flag_pre',
        #    dt=defaultclock.dt,
        #    when='after_resets',
        #    order=1)
        #post_neurons.run_regularly(
        #    'Ca = Ca*int(Ca>0) - Ca*int(Ca<0)',
        #    name='clear_spike_flag_post',
        #    dt=defaultclock.dt,
        #    when='after_resets',
        #    order=1)

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
        ref_statemon_pre_neurons = StateMonitor(ref_pre_neurons,
                                                variables=['Vm'],
                                                record=range(N_pre),
                                                name='ref_statemon_pre_neurons')
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
            import pdb;pdb.set_trace()
            print(statemon_post_synapse.w_plast[0][:100])
            plot_delta_ti = 0
            plot_delta_tf = 400
            plt.scatter(spikemon_pre_neurons.t/ms, spikemon_pre_neurons.i)
            plt.scatter(spikemon_post_neurons.t/ms, spikemon_post_neurons.i)
            plt.show()

            plt.clear_figure()
            plt.plot(statemon_post_synapse.t/ms,
                     aux_plot(statemon_post_synapse.w_plast[0]))
            plt.show()

            plt.clear_figure()
            plt.plot(statemon_post_neurons.t/ms,
                     aux_plot_Ca(statemon_post_neurons.Ca[0][plot_delta_ti:plot_delta_tf]))
            plt.plot(statemon_pre_neurons.t/ms,
                     aux_plot_Ca(statemon_pre_neurons.Ca[0][plot_delta_ti:plot_delta_tf]))
            plt.show()

            plt.clear_figure()
            max_val = max(aux_plot(statemon_pre_neurons.Vm[0]))
            norm_vm = [x/max_val
                           for x in aux_plot(statemon_pre_neurons.Vm[0])]
            plt.plot(statemon_pre_neurons.t[plot_delta_ti:plot_delta_tf]/ms, norm_vm[plot_delta_ti:plot_delta_tf])
            max_val = max(ref_statemon_pre_neurons.Vm[0])
            norm_ref_vm = [x/max_val for x in ref_statemon_pre_neurons.Vm[0]]
            plt.plot(ref_statemon_pre_neurons.t[plot_delta_ti:plot_delta_tf]/ms, norm_ref_vm[plot_delta_ti:plot_delta_tf])
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
