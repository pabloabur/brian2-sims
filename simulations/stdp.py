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
import quantities as q
import sys
import json

from brian2 import run, device, defaultclock, scheduling_summary,\
    SpikeGeneratorGroup, StateMonitor, SpikeMonitor, TimedArray, PoissonInput,\
    prefs, EventMonitor
from brian2 import mV, second, Hz, ms

import plotext as plt

from core.utils.misc import minifloat2decimal, decimal2minifloat
from core.utils.process_responses import neurons_rate, statemonitors2dataframe
from core.utils.prepare_models import generate_connection_indices,\
    set_hardwarelike_scheme

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


def stimuli_protocol1():
    """Stimulus gneration for STDP protocols.

    This function returns two brian2 objects.
    Both are Spikegeneratorgroups which hold a single index each
    and varying spike times.
    The protocol follows homoeostasis, weak LTP, weak LTD, strong LTP,
    strong LTD, homoeostasis.

    Returns:
        SpikeGeneratorGroup (brian2.obj: Brian2 objects which hold the
            spiketimes and the respective neuron indices.
    """
    t_pre_homoeotasis_1 = np.linspace(0, 300, 5)
    t_pre_weakLTP = np.linspace(t_pre_homoeotasis_1[-1] + 100,
                                t_pre_homoeotasis_1[-1] + 400,
                                5)
    t_pre_weakLTD = np.linspace(t_pre_weakLTP[-1] + 100,
                                t_pre_weakLTP[-1] + 400,
                                5)
    t_pre_strongLTP = np.linspace(t_pre_weakLTD[-1] + 100,
                                  t_pre_weakLTD[-1] + 400,
                                  5)
    t_pre_strongLTD = np.linspace(t_pre_strongLTP[-1] + 100,
                                  t_pre_strongLTP[-1] + 400,
                                  5)
    t_pre_homoeotasis_2 = np.linspace(t_pre_strongLTD[-1] + 100,
                                      t_pre_strongLTD[-1] + 400,
                                      5)
    t_pre = np.hstack((t_pre_homoeotasis_1, t_pre_weakLTP, t_pre_weakLTD,
                       t_pre_strongLTP, t_pre_strongLTD, t_pre_homoeotasis_2))

    # Normal distributed shift of spike times to ensure homoeotasis
    t_post_homoeotasis_1 = t_pre_homoeotasis_1 + \
        np.random.randint(-3, 3, len(t_pre_homoeotasis_1))
    t_post_weakLTP = t_pre_weakLTP + 15   # post spikes some ms after pre
    t_post_weakLTD = t_pre_weakLTD - 15   # post spikes some ms before pre
    t_post_strongLTP = t_pre_strongLTP + 3  # post spike some ms after pre
    t_post_strongLTD = t_pre_strongLTD - 3  # and some ms before pre
    t_post_homoeotasis_2 = t_pre_homoeotasis_2 + \
        np.random.randint(-3, 3, len(t_pre_homoeotasis_2))

    t_post = np.hstack((t_post_homoeotasis_1, t_post_weakLTP, t_post_weakLTD,
                        t_post_strongLTP, t_post_strongLTD,
                        t_post_homoeotasis_2))
    t_post = np.clip(t_post, 0, np.inf)
    ind_pre = np.zeros(len(t_pre))
    ind_post = np.zeros(len(t_post))

    pre = SpikeGeneratorGroup(
        1, indices=ind_pre, times=t_pre * ms, name='gPre')
    post = SpikeGeneratorGroup(
        1, indices=ind_post, times=t_post * ms, name='gPost')

    tmax = max(pre._spike_time[-1], post._spike_time[-1]) * second
    return pre, post, tmax


def stimuli_protocol2():
    trials = 20
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
    rng = np.random.default_rng()
    run_namespace = {}

    """ ================ models and helper functions ================ """
    if args.precision == 'fp8':
        neuron_model = fp8LIF()
        def aux_w_sample(x): return decimal2minifloat(x, raise_warning=False)
        def aux_plot(x): return minifloat2decimal(x)
        def aux_plot_Ca(x): return minifloat2decimal(x)
        synapse_model = fp8CUBA()
        stdp_model = fp8STDP()
    elif args.precision == 'fp64':
        neuron_model = tsvLIF()
        def aux_w_sample(x): return x * mV
        def aux_plot(x): return x/mV
        def aux_plot_Ca(x): return x
        synapse_model = tsvCUBA()
        stdp_model = tsvSTDP()
    else:
       raise UserWarning('Precision not supported')

    ref_neuron_model = LIF()
    ref_synapse_model = CUBA()
    ref_stdp_model = STDP()

    neuron_model.modify_model('events', args.event_condition, key='active_Ca',)

    """ ================ Protocol specifications ================ """
    if args.protocol == 1:
        N_pre, N_post = 10, 10
        n_conns = N_pre
        sampled_weights = [11 for _ in range(n_conns)]
        static_weight = 120
        conn_condition = 'i==j'

        ref_neuron_model.modify_model('model',
                                      'gtot = gtot0 + gtot1',
                                      old_expr='gtot = gtot0')
        ref_neuron_model.model += 'gtot1 : volt\n'

        pre_spikegenerator, post_spikegenerator, tmax = stimuli_protocol1()
        pre_neurons = create_neurons(N_pre, neuron_model)
        ref_pre_neurons = create_neurons(N_pre, ref_neuron_model)
        post_neurons = create_neurons(N_post, neuron_model)
        ref_post_neurons = create_neurons(N_post, ref_neuron_model)

        synapse_model.modify_model('parameters',
                                   aux_w_sample(static_weight),
                                   key='weight')
        ref_synapse_model.modify_model('parameters',
                                       static_weight*mV,
                                       key='weight')
        pre_synapse = create_synapses(pre_spikegenerator,
                                      pre_neurons,
                                      synapse_model,
                                      name='static_pre_synapse')
        ref_pre_synapse = create_synapses(pre_spikegenerator,
                                          ref_pre_neurons,
                                          ref_synapse_model,
                                          name='ref_static_pre_synapse')
        post_synapse = create_synapses(post_spikegenerator,
                                       post_neurons,
                                       synapse_model,
                                       name='static_post_synapse')
        ref_post_synapse = create_synapses(post_spikegenerator,
                                           ref_post_neurons,
                                           ref_synapse_model,
                                           name='ref_static_post_synapse')

        stdp_model.modify_model('connection',
                                conn_condition,
                                key='condition')
        ref_stdp_model.modify_model('connection',
                                    conn_condition,
                                    key='condition')

        ref_stdp_model.modify_model('model', 'gtot1_post', old_expr='gtot0_post')

    elif args.protocol == 2:
        tapre, tapost, tmax, N, trial_duration = stimuli_protocol2()
        N_pre, N_post = N, N
        n_conns = N
        conn_condition = 'i==j'

        sampled_weights = [50 for _ in range(n_conns)]
        # TODO for fp8? The idea is to inject current so neurons spikes, but maybe I dont have to
        #neuron_model.modify_model(
        #    'model',
        #    'summed_decay = tapre(t, i)',
        #    old_expr='summed_decay = fp8_add(decay_term, gtot*int(not_refractory))')
        neuron_model.modify_model('threshold',
                                  'tapre(t, i) == 1',
                                  old_expr='Vm > Vthr')
        pre_neurons = create_neurons(N, neuron_model)

        #neuron_model.modify_model(
        #    'model',
        #    'summed_decay = tapost(t, i)',
        #    old_expr='summed_decay = fp8_add(decay_term, gtot*int(not_refractory))')
        neuron_model.modify_model('threshold', 'tapost', old_expr='tapre')
        post_neurons = create_neurons(N, neuron_model)

        stdp_model.modify_model('connection',
                                conn_condition,
                                key='condition')

        tmax = tmax * ms

        run_namespace.update({'tapre': tapre, 'tapost': tapost})

    elif args.protocol == 3:
        N_pre = 1000
        N_post = 1
        n_conns = N_pre
        sampled_weights = rng.gamma(.1, 1.75, n_conns)
        tmax = 10000 * ms
        conn_condition = None

        post_neurons = create_neurons(N_post, neuron_model)
        neuron_model.modify_model('threshold', 'rand()<rates*dt')
        neuron_model.model += 'rates : Hz\n'
        neuron_model.modify_model('parameters', rng.uniform(5, 15, N_pre)*ms, key='tau_ca')
        pre_neurons = create_neurons(N_pre, neuron_model)
        pre_neurons.rates = 15*Hz

        stdp_model.modify_model('connection',
                                conn_condition,
                                key='condition')

    """ ================ General specifications ================ """
    stdp_model.modify_model('parameters',
                            aux_w_sample(sampled_weights),
                            key='w_plast')
    ref_stdp_model.modify_model('parameters',
                                sampled_weights*mV,
                                key='w_plast')
    stdp_model.modify_model('namespace', 0.1*mV, key='eta')
    ref_stdp_model.modify_model('namespace', 0.1*mV, key='eta')

    ref_stdp_model.modify_model('on_pre',
                                'int(lastspike_post!=lastspike_pre)*eta*j_trace',
                                old_expr='eta*j_trace')
    ref_stdp_model.modify_model('on_post',
                                'int(lastspike_post!=lastspike_pre)*eta*i_trace',
                                old_expr='eta*i_trace')

    stdp_synapse = create_synapses(pre_neurons,
                                   post_neurons,
                                   stdp_model,
                                   name='stdp_synapse')
    ref_stdp_synapse = create_synapses(ref_pre_neurons,
                                       ref_post_neurons,
                                       ref_stdp_model,
                                       name='ref_stdp_synapse')

    neurons_list = [pre_neurons, post_neurons]
    set_hardwarelike_scheme(prefs, neurons_list, defaultclock.dt)

    """ ================ Setting up monitors ================ """
    spikemon_pre_neurons = SpikeMonitor(pre_neurons,
                                        name='spikemon_pre_neurons')
    spikemon_post_neurons = SpikeMonitor(post_neurons,
                                         name='spikemon_post_neurons')
    if args.protocol < 3:
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
    ref_statemon_post_neurons = StateMonitor(ref_post_neurons,
                                         variables=['Vm'],
                                         record=range(N_post),
                                         name='ref_statemon_post_neurons')
    statemon_post_synapse = StateMonitor(stdp_synapse,
                                        variables=['w_plast'],
                                        record=range(n_conns),
                                        name='statemon_post_synapse')
    ref_statemon_post_synapse = StateMonitor(ref_stdp_synapse,
                                        variables=['w_plast'],
                                        record=range(n_conns),
                                        name='ref_statemon_post_synapse')
    active_monitor = EventMonitor(pre_neurons, 'active_Ca', 'Ca')

    run(tmax, report='stdout', namespace=run_namespace)

    if args.backend == 'cpp_standalone' or args.backend == 'cuda_standalone':
        device.build(args.code_path)

    """ =================== Saving data =================== """
    metadata = {'event_condition': args.event_condition
                }
    with open(f'{args.save_path}/metadata.json', 'w') as f:
        json.dump(metadata, f)

    output_spikes = pd.DataFrame(
        {'time_ms': np.array(active_monitor.t/defaultclock.dt),
         'id': np.array(active_monitor.i)})
    output_spikes.to_csv(f'{args.save_path}/events_spikes.csv', index=False)

    output_vars = statemonitors2dataframe([statemon_post_synapse,
                                           ref_statemon_post_synapse])
    output_vars.to_csv(f'{args.save_path}/synapse_vars.csv', index=False)

    if args.protocol < 3:
        output_vars = statemonitors2dataframe([statemon_pre_neurons,
                                           statemon_post_neurons,
                                           ref_statemon_pre_neurons,
                                           ref_statemon_post_neurons])
        output_vars.to_csv(f'{args.save_path}/state_vars.csv', index=False)

    if not args.quiet:
        if args.protocol == 1:
            delta_ti = 0
            delta_tf = -1
            plt.scatter(spikemon_pre_neurons.t[delta_ti:delta_tf]/ms,
                        spikemon_pre_neurons.i[delta_ti:delta_tf])
            plt.scatter(spikemon_post_neurons.t[delta_ti:delta_tf]/ms,
                        spikemon_post_neurons.i[delta_ti:delta_tf])
            plt.build()
            plt.title('Spikes')
            plt.save_fig(f'{args.save_path}/fig1.txt', keep_colors=True)

            plt.clear_figure()
            plt.plot(statemon_post_neurons.t[delta_ti:delta_tf]/ms,
                     aux_plot_Ca(
                         statemon_post_neurons.Ca[0][delta_ti:delta_tf]))
            plt.plot(statemon_pre_neurons.t[delta_ti:delta_tf]/ms,
                     aux_plot_Ca(
                         statemon_pre_neurons.Ca[0][delta_ti:delta_tf]))
            plt.title('Time window evolution')
            plt.build()
            plt.save_fig(f'{args.save_path}/fig3.txt', keep_colors=True)

            plt.clear_figure()
            plt.title('difference between Vms')
            plt.plot(ref_statemon_post_neurons.t/ms, ref_statemon_post_neurons.Vm[0]/mV, label='ref')
            plt.plot(statemon_post_neurons.t/ms, statemon_post_neurons.Vm[0]/mV, label='base')
            plt.build()
            plt.save_fig(f'{args.save_path}/fig4.txt', keep_colors=True)

            plt.clear_figure()
            plt.title('difference between weights')
            plt.plot(ref_statemon_post_synapse.t/ms, ref_statemon_post_synapse.w_plast[0]/mV, label='ref')
            plt.plot(statemon_post_synapse.t/ms, statemon_post_synapse.w_plast[0]/mV, label='base')
            plt.build()
            plt.save_fig(f'{args.save_path}/fig5.txt', keep_colors=True)

            plt.clear_figure()
            plt.subplots(2, 1)
            plt.subplot(1, 1).title('Weight mean square error')
            plt.plot(((statemon_post_synapse.w_plast/mV - ref_statemon_post_synapse.w_plast/mV)**2).mean(axis=0), label='MSE')
            plt.subplot(2, 1).title('Vm mean square error')
            plt.plot(((statemon_post_neurons.Vm/mV - ref_statemon_post_neurons.Vm/mV)**2).mean(axis=0), label='MSE')
            plt.build()
            plt.save_fig(f'{args.save_path}/fig6.txt', keep_colors=True)


        elif args.protocol == 2:
            pairs_timing = (spikemon_post_neurons.t[:trial_duration]
                            - spikemon_post_neurons.t[:trial_duration][::-1])/ms
            plt.plot(pairs_timing,
                     aux_plot(statemon_post_synapse.w_plast[:, -1]))
            plt.build()
            plt.save_fig(f'{args.save_path}/fig1.txt')

            plt.clear_figure()
            plt.plot(aux_plot_Ca(statemon_pre_neurons.Ca[0][:100]))
            plt.plot(aux_plot_Ca(statemon_post_neurons.Ca[0][:100]))
            plt.build()
            plt.save_fig(f'{args.save_path}/fig2.txt')

            plt.clear_figure()
            plt.plot(aux_plot_Ca(statemon_pre_neurons.Ca[29][:100]))
            plt.plot(aux_plot_Ca(statemon_post_neurons.Ca[29][:100]))
            plt.build()
            plt.save_fig(f'{args.save_path}/fig3.txt')

            plt.clear_figure()
            plt.scatter(spikemon_pre_neurons.t[:150]/ms, spikemon_pre_neurons.i[:150])
            plt.scatter(spikemon_post_neurons.t[:150]/ms, spikemon_post_neurons.i[:150])
            plt.build()
            plt.save_fig(f'{args.save_path}/fig4.txt')

        elif args.protocol == 3:
            plt.hist(aux_plot(statemon_post_synapse.w_plast[:, -1]))
            plt.build()
            plt.save_fig(f'{args.save_path}/fig1.txt')

            plt.clear_figure()
            plt.plot(aux_plot(statemon_post_synapse.w_plast[0, :]))
            plt.plot(aux_plot(statemon_post_synapse.w_plast[500, :]))
            plt.plot(aux_plot(statemon_post_synapse.w_plast[999, :]))
            plt.build()
            plt.save_fig(f'{args.save_path}/fig2.txt')

            neu_r = neurons_rate(spikemon_pre_neurons, tmax/ms)
            plt.clear_figure()
            plt.plot(neu_r.times/q.ms, neu_r[:, 1].magnitude.flatten())
            plt.plot(neu_r.times/q.ms, neu_r[:, 2].magnitude.flatten())
            plt.build()
            plt.save_fig(f'{args.save_path}/fig3.txt')

            neu_r = neurons_rate(spikemon_post_neurons, tmax/ms)
            plt.clear_figure()
            plt.plot(neu_r.times/q.ms, neu_r.magnitude.flatten())
            plt.build()
            plt.save_fig(f'{args.save_path}/fig4.txt')
