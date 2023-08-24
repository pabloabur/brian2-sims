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

from brian2 import run, device, defaultclock, scheduling_summary,\
    SpikeGeneratorGroup, StateMonitor, SpikeMonitor, TimedArray, PoissonInput,\
    prefs, EventMonitor
from brian2 import mV, second, Hz, ms

import plotext as plt

from core.utils.misc import minifloat2decimal, decimal2minifloat
from core.utils.process_responses import neurons_rate
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

    """ ================ Protocol specifications ================ """
    if args.protocol == 1:
        N_pre, N_post = 2, 2
        n_conns = N_pre
        sampled_weights = [11 for _ in range(n_conns)]
        static_weight = 120
        conn_condition = 'i==j'

        pre_spikegenerator, post_spikegenerator, tmax = stimuli_protocol1()
        pre_neurons = create_neurons(2, neuron_model)
        post_neurons = create_neurons(2, neuron_model)

        synapse_model.modify_model('parameters',
                                   aux_w_sample(static_weight),
                                   key='weight')
        pre_synapse = create_synapses(pre_spikegenerator,
                                      pre_neurons,
                                      synapse_model,
                                      name='static_pre_synapse')
        post_synapse = create_synapses(post_spikegenerator,
                                       post_neurons,
                                       synapse_model,
                                       name='static_post_synapse')

        stdp_model.modify_model('connection',
                                conn_condition,
                                key='condition')

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

    elif args.protocol == 4:
        tmax = 60000 * ms
        Ne, Ni = 900, 225
        Nt = Ne + Ni
        exc_delay = 1.5
        ext_weights = .25
        exc_weights = 4.561
        inh_weights = 5 * exc_weights
        p_conn = .1


        neuron_model.modify_model('parameters',
                                  'clip(5.7 + randn()*7.2, 0, inf)*mV',
                                  key='Vm')

        neurons = create_neurons(Ne+Ni,
                                 neuron_model)
        exc_neurons = neurons[:Ne]
        inh_neurons = neurons[Ne:]
        # N.B. relative to pre and post populations of stdp_synapse
        pre_neurons, post_neurons = exc_neurons, neurons
        N_pre, N_post = Ne, Nt

        poisson_spikes = PoissonInput(neurons, 'g', 9000, rate=2.32*Hz,
                                      weight=f'{ext_weights}*mV')

        tmp_eq = f'int(clip({exc_delay} + randn()*{exc_delay/2}, 1, inf))*ms'
        stdp_model.modify_model('parameters', tmp_eq, key='delay')
        synapse_model.modify_model('parameters', tmp_eq, key='delay')

        sources_e, targets_e = generate_connection_indices(Ne, Nt, p_conn,
                                                           allow_autapse=False)
        stdp_model.modify_model('connection', sources_e, key='i')
        stdp_model.modify_model('connection', targets_e, key='j')
        sources_i, targets_i = generate_connection_indices(Ni, Nt, p_conn,
                                                           allow_autapse=False)
        synapse_model.modify_model('connection', sources_i, key='i')
        synapse_model.modify_model('connection', targets_i, key='j')

        tmp_eq = f'clip({inh_weights} + randn()*{inh_weights/10}, 0, inf)*mV'
        synapse_model.modify_model('parameters', tmp_eq, key='weight')
        synapse_model.modify_model('namespace', -1, key='w_factor')

        n_conns = len(sources_e)
        sampled_weights = np.clip(rng.normal(exc_weights,
                                             exc_weights/10,
                                             n_conns),
                                  0, np.inf)

        # Inhibitory group, called `pre_synapse` for compatibility with other
        # protocols
        pre_synapse = create_synapses(inh_neurons,
                                      neurons,
                                      synapse_model)

        run_namespace.update({'ext_weights': ext_weights})

    """ ================ General specifications ================ """
    stdp_model.modify_model('parameters',
                            aux_w_sample(sampled_weights),
                            key='w_plast')
    stdp_model.modify_model('namespace', 0.1*mV, key='eta')

    stdp_synapse = create_synapses(pre_neurons,
                                   post_neurons,
                                   stdp_model,
                                   name='stdp_synapse')

    if args.precision == 'fp64':
        if args.protocol == 4:
            neurons_list = [post_neurons]
        else:
            neurons_list = [pre_neurons, post_neurons]
        set_hardwarelike_scheme(prefs, neurons_list, defaultclock.dt)

    """ ================ Setting up monitors ================ """
    if args.protocol != 4:
        spikemon_pre_neurons = SpikeMonitor(pre_neurons,
                                            name='spikemon_pre_neurons')
    spikemon_post_neurons = SpikeMonitor(post_neurons,
                                         name='spikemon_post_neurons')
    if args.protocol < 3:
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
                                        record=range(n_conns),
                                        name='statemon_post_synapse')
    active_monitor = EventMonitor(pre_neurons, 'active_Ca', 'Ca')

    run(tmax, report='stdout', namespace=run_namespace)

    if args.backend == 'cpp_standalone' or args.backend == 'cuda_standalone':
        device.build(args.code_path)

    """ =================== Saving data =================== """
    output_spikes = pd.dataframe(
        {'time_ms': np.array(spikemon_post_neurons.t/defaultclock.dt),
         'id': np.array(spikemon_post_neurons.i)})
    output_spikes.to_csv(f'{args.save_path}/post_spikes.csv', index=false)
    if args.protocol != 4:
        output_spikes = pd.dataframe(
                {'time_ms': np.array(spikemon_pre_neurons.t/defaultclock.dt),
                 'id': np.array(spikemon_pre_neurons.i)})
        output_spikes.to_csv(f'{args.save_path}/post_spikes.csv', index=false)

    output_vars = statemonitors2dataframe([statemon_post_synapse,
                                           active_monitor])
    output_vars.to_csv(f'{args.save_path}/synapse_vars.csv', index=False)

    if args.protocol < 3:
        output_vars = statemonitors2dataframe([statemon_post_neurons,
                                               statemon_pre_neurons])
        output_vars.to_csv(f'{args.save_path}/neuron_vars.csv', index=False)

    if not args.quiet:
        if args.protocol == 1:
            delta_ti = 0
            delta_tf = -1
            plt.scatter(spikemon_pre_neurons.t[delta_ti:delta_tf]/ms,
                        spikemon_pre_neurons.i[delta_ti:delta_tf])
            plt.scatter(spikemon_post_neurons.t[delta_ti:delta_tf]/ms,
                        spikemon_post_neurons.i[delta_ti:delta_tf])
            plt.build()
            plt.save_fig(f'{args.save_path}/fig1.txt')

            plt.clear_figure()
            plt.plot(statemon_post_synapse.t[delta_ti:delta_tf]/ms,
                     aux_plot(
                         statemon_post_synapse.w_plast[0][delta_ti:delta_tf]))
            plt.build()
            plt.save_fig(f'{args.save_path}/fig2.txt')

            plt.clear_figure()
            plt.plot(statemon_post_neurons.t[delta_ti:delta_tf]/ms,
                     aux_plot_Ca(
                         statemon_post_neurons.Ca[0][delta_ti:delta_tf]))
            plt.plot(statemon_pre_neurons.t[delta_ti:delta_tf]/ms,
                     aux_plot_Ca(
                         statemon_pre_neurons.Ca[0][delta_ti:delta_tf]))
            plt.build()
            plt.save_fig(f'{args.save_path}/fig3.txt')

            plt.clear_figure()
            plt.plot(statemon_pre_neurons.t[delta_ti:delta_tf]/ms,
                     aux_plot(statemon_pre_neurons.Vm[0]))
            plt.build()
            plt.save_fig(f'{args.save_path}/fig4.txt')

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

        elif args.protocol == 4:
            num_fetches = {'pre': spikemon_post_neurons.num_spikes,
                           'fanout': active_monitor.num_events}
            print(f'Potential memory fetches for each strategy:\nConventional:'
                  f' {2*num_fetches["pre"]/1e6}M\nFanout: '
                  f'{num_fetches["fanout"]/1e6}M')

            max_weight_idx = np.where(
                stdp_synapse.w_plast == max(stdp_synapse.w_plast))[0]
            target_id = stdp_synapse.j[max_weight_idx[0]]
            plt.clear_figure()
            plt.hist(np.array(stdp_synapse.w_plast/mV)[stdp_synapse.j==target_id])
            plt.title('Weights targetting a neuron')
            plt.build()
            plt.save_fig(f'{args.save_path}/fig1.txt')

            source_ids = np.array(stdp_synapse.i)[stdp_synapse.j==target_id]
            n_incoming = np.shape(
                statemon_post_synapse.w_plast[stdp_synapse.j==target_id, :])[0]
            plt.clear_figure()
            for x in range(n_incoming):
                plt.plot(np.array(
                    statemon_post_synapse.w_plast/mV)[stdp_synapse.j==target_id,
                                                      :][x, :])
            plt.title(f'Neurons {source_ids} targeting {target_id} over time')
            plt.build()
            plt.save_fig(f'{args.save_path}/fig2.txt')

            plt.clear_figure()
            plt.hist(stdp_synapse.w_plast/mV)
            plt.title('Final distribution of weights')
            plt.build()
            plt.save_fig(f'{args.save_path}/fig3.txt')
