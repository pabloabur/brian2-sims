""" This file has modified versions of some code from Teili. See
M. Milde, A. Renner, R. Krause, A. M. Whatley, S. Solinas, D. Zendrikov,
N. Risi, M. Rasetto, K. Burelo, V. R. C. Leite. teili: A toolbox for building
and testing neural algorithms and computational primitives using spiking
neurons. Unreleased software, Institute of Neuroinformatics, University of
Zurich and ETH Zurich, 2018.
"""

import numpy as np
import pandas as pd
import quantities as q
import json
import feather

from brian2 import run, device, defaultclock, scheduling_summary,\
    SpikeGeneratorGroup, StateMonitor, SpikeMonitor, TimedArray, PoissonInput,\
    prefs, EventMonitor
from brian2 import mV, second, Hz, ms

import plotext as plt

from core.equations.base_equation import ParamDict
from core.utils.misc import minifloat2decimal, decimal2minifloat
from core.utils.process_responses import neurons_rate, statemonitors2dataframe
from core.utils.prepare_models import generate_connection_indices,\
    set_hardwarelike_scheme

from core.equations.neurons.fp8LIF import fp8LIF
from core.equations.synapses.fp8CUBA import fp8CUBA
from core.equations.synapses.fp8STDP import fp8STDP
from core.equations.neurons.sfp8LIF import sfp8LIF
from core.equations.synapses.sfp8CUBA import sfp8CUBA
from core.equations.synapses.sfp8STDP import sfp8STDP
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
    """Stimulus generation for STDP protocols.

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
    trials = 60
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
        neuron_model = sfp8LIF()
        def aux_w_sample(x): return decimal2minifloat(x, raise_warning=False)
        def aux_plot(x): return minifloat2decimal(x)
        def aux_plot_Ca(x): return minifloat2decimal(x)
        def aux_plot_Vm(x): return minifloat2decimal(x)
        synapse_model = sfp8CUBA()
        stdp_model = sfp8STDP()
    elif args.precision == 'fp64':
        neuron_model = tsvLIF()
        def aux_w_sample(x): return x * mV
        def aux_plot(x): return x/mV
        def aux_plot_Ca(x): return x
        def aux_plot_Vm(x): return x/mV
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
        pre_spikegenerator, post_spikegenerator, tmax = stimuli_protocol1()
        w_mon_dt = defaultclock.dt

        # for stochastic case, more neurons are necessary for average
        conn_condition = 'i==j'
        if args.precision == 'fp8':
            N_pre, N_post = 100, 100
            n_conns = N_pre
            static_weight = 176
            w_init = args.w_init
            sampled_weights = [w_init for _ in range(n_conns)]
        else:
            N_pre, N_post = 10, 10
            n_conns = N_pre
            static_weight = 120
            w_init = 11
            sampled_weights = [w_init for _ in range(n_conns)]

        ref_neuron_model.modify_model('model',
                                      'gtot = gtot0 + gtot1',
                                      old_expr='gtot = gtot0')
        ref_neuron_model.model += 'gtot1 : volt\n'

        pre_neurons = create_neurons(N_pre, neuron_model)
        ref_pre_neurons = create_neurons(N_pre, ref_neuron_model)
        post_neurons = create_neurons(N_post, neuron_model)
        ref_post_neurons = create_neurons(N_post, ref_neuron_model)

        ref_synapse_model.modify_model('parameters',
                                       static_weight*mV,
                                       key='weight')

        synapse_model.modify_model('parameters',
                                   aux_w_sample(static_weight),
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
        ref_stdp_source = ref_pre_neurons

    elif args.protocol == 2:
        tapre, tapost, tmax, N, trial_duration = stimuli_protocol2()
        w_mon_dt = defaultclock.dt

        N_pre, N_post = N, N
        n_conns = N
        conn_condition = 'i==j'
        w_init = args.w_init
        sampled_weights = [w_init for _ in range(n_conns)]

        # TODO for fp8? The idea is to inject current so neurons spikes, but maybe I dont have to
        #neuron_model.modify_model(
        #    'model',
        #    'summed_decay = tapre(t, i)',
        #    old_expr='summed_decay = fp8_add(decay_term, gtot*int(not_refractory))')
        if args.precision == 'fp8':
            old_expr = 'Vm == Vthr'
        else:
            old_expr = 'Vm > Vthr'
        neuron_model.modify_model('threshold',
                                  'tapre(t, i) == 1',
                                  old_expr=old_expr)
        ref_neuron_model.modify_model('threshold',
                                      'tapre(t, i) == 1',
                                      old_expr='Vm > Vthr')

        pre_neurons = create_neurons(N, neuron_model)
        ref_pre_neurons = create_neurons(N, ref_neuron_model)

        neuron_model.modify_model('threshold', 'tapost', old_expr='tapre')
        ref_neuron_model.modify_model('threshold', 'tapost', old_expr='tapre')
        post_neurons = create_neurons(N, neuron_model)
        ref_post_neurons = create_neurons(N, ref_neuron_model)

        stdp_model.modify_model('connection',
                                conn_condition,
                                key='condition')
        ref_stdp_model.modify_model('connection',
                                     conn_condition,
                                     key='condition')
        ref_stdp_source = ref_pre_neurons

        tmax = tmax * ms

        run_namespace.update({'tapre': tapre, 'tapost': tapost})

    elif args.protocol == 3:
        N_pre = 1000
        N_post = args.N_post
        n_conns = N_pre
        w_init = 0.3
        sampled_weights = w_init # rng.gamma(1, 17.5, n_conns)
        tmax = args.tmax * ms
        conn_condition = None
        w_mon_dt = 1000 * ms

        post_neurons = create_neurons(N_post, neuron_model)
        ref_post_neurons = create_neurons(N_post, ref_neuron_model)

        neuron_model.model += 'event_count : 1\n'
        mon_active_neurons = create_neurons(1, neuron_model)
        mon_active_neurons.run_regularly(
            'event_count = 0',
            name=f'clear_event_counter',
            dt=1*ms,
            when='after_synapses',
            order=1)

        neuron_model.modify_model('threshold', 'rand()<rates*dt')
        neuron_model.model += 'rates : Hz\n'
        neuron_model.modify_model('parameters', '(10*rand() + 18)*ms', #  '(10*rand() + 5)*ms'
                                  key='tau_ca')
        pre_neurons = create_neurons(N_pre, neuron_model)
        pre_neurons.rates = 15*Hz

        stdp_model.modify_model('connection',
                                conn_condition,
                                key='condition')
        ref_stdp_model.modify_model('model', 'tau_syn_ref',
                                    old_expr='tau_syn')
        ref_stdp_model.modify_model('model', 'alpha_syn_ref',
                                    old_expr='alpha_syn')
        ref_stdp_model.modify_model('parameters', 13*ms,
                                    key='tau_itrace')
        ref_stdp_model.modify_model('parameters', 28*ms,
                                    key='tau_jtrace')
        ref_stdp_model.modify_model('on_post', 'j_trace += 1.05',
                                    old_expr='j_trace += 1')
        ref_stdp_model.modify_model('on_pre',
                                    'g_syn += ',
                                    old_expr='g += ')
        ref_stdp_model.modify_model('model',
                                    f'dg_syn/dt = alpha_syn_ref*g_syn/second '
                                    f': volt (clock-driven)',
                                    old_expr=f'dg/dt = alpha_syn_ref*g/second '
                                             f': volt (clock-driven)')
        ref_stdp_model.modify_model('model',
                                    'gtot0_post = g_syn*w_factor : volt (summed)',
                                    old_expr='gtot0_post = g*w_factor : volt (summed)')
        ref_stdp_model.parameters = ParamDict({**ref_stdp_model.parameters,
                                               **{'tau_syn_ref':  '5*ms'}})
        del ref_stdp_model.parameters['tau_syn']
        ref_stdp_model.parameters = ParamDict({**ref_stdp_model.parameters,
                                               **{'alpha_syn_ref': f'tau_syn_ref'
                                                                   f'/(dt + tau_syn_ref)'
                                                  }})
        del ref_stdp_model.parameters['alpha_syn']
        ref_stdp_model.modify_model('connection',
                                     conn_condition,
                                     key='condition')
        ref_stdp_source = pre_neurons

    """ ================ General specifications ================ """
    if args.precision == 'fp64':
        stdp_model.modify_model('namespace', args.w_max*mV, key='w_max')
    stdp_model.modify_model('parameters',
                            aux_w_sample(sampled_weights),
                            key='w_plast')
    ref_stdp_model.modify_model('namespace', args.w_max*mV, key='w_max')
    ref_stdp_model.modify_model('parameters',
                                sampled_weights*mV,
                                key='w_plast')
    if args.precision == 'fp64':
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
    ref_stdp_synapse = create_synapses(ref_stdp_source,
                                       ref_post_neurons,
                                       ref_stdp_model,
                                       name='ref_stdp_synapse')

    neurons_list = [pre_neurons, post_neurons]
    set_hardwarelike_scheme(prefs, neurons_list, defaultclock.dt, args.precision)

    """ ================ Setting up monitors ================ """
    spikemon_pre_neurons = SpikeMonitor(pre_neurons,
                                        name='spikemon_pre_neurons')
    spikemon_post_neurons = SpikeMonitor(post_neurons,
                                         name='spikemon_post_neurons')
    ref_spikemon_post_neurons = SpikeMonitor(ref_post_neurons,
                                             name='ref_spikemon_post_neurons')
    if args.protocol < 3:
        statemon_pre_neurons = StateMonitor(pre_neurons,
                                            variables=['Vm', 'g', 'Ca'],
                                            record=range(N_pre),
                                            name='statemon_pre_neurons')
        ref_statemon_pre_neurons = StateMonitor(ref_pre_neurons,
                                            variables=['Vm'],
                                            record=range(N_pre),
                                            name='ref_statemon_pre_neurons')
    elif args.protocol == 3:
        stdp_model.on_pre['stdp_fanout'] = f'''
            event_count_post += 1
            '''
        stdp_model.modify_model('connection', 1., key='p')
        stdp_mon_events = create_synapses(pre_neurons, mon_active_neurons, stdp_model)
        active_monitor = StateMonitor(mon_active_neurons,
                                      'event_count',
                                      record=0,
                                      when='after_synapses',
                                      order=0,
                                      name='event_count_monitor')
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
                                        dt=w_mon_dt,
                                        name='statemon_post_synapse')
    ref_statemon_post_synapse = StateMonitor(ref_stdp_synapse,
                                        variables=['w_plast', 'i_trace',  'j_trace'],
                                        dt=w_mon_dt,
                                        record=range(n_conns),
                                        name='ref_statemon_post_synapse')

    run(tmax, report='stdout', namespace=run_namespace)

    if args.backend == 'cpp_standalone' or args.backend == 'cuda_standalone':
        device.build(args.code_path)

    """ =================== Saving data =================== """
    metadata = {'event_condition': args.event_condition,
                'N_post': N_post,
                'stochastic_rounding': True,
                'w_init': w_init
                }
    with open(f'{args.save_path}/metadata.json', 'w') as f:
        json.dump(metadata, f)

    output_spikes = pd.DataFrame(
        {'time_ms': np.array(spikemon_pre_neurons.t/defaultclock.dt),
         'id': np.array(spikemon_pre_neurons.i)}
    )
    output_spikes.to_csv(f'{args.save_path}/spikes_pre.csv', index=False)
    # do the same for post and ref
    output_spikes = pd.DataFrame(
        {'time_ms': np.array(spikemon_post_neurons.t/defaultclock.dt),
         'id': np.array(spikemon_post_neurons.i)}
    )
    output_spikes.to_csv(f'{args.save_path}/spikes_post.csv', index=False)
    output_spikes = pd.DataFrame(
        {'time_ms': np.array(ref_spikemon_post_neurons.t/defaultclock.dt),
         'id': np.array(ref_spikemon_post_neurons.i)}
    )
    output_spikes.to_csv(f'{args.save_path}/ref_spikes_post.csv', index=False)

    output_vars = statemonitors2dataframe([statemon_post_synapse,
                                           ref_statemon_post_synapse])
    output_vars.to_csv(f'{args.save_path}/synapse_vars.csv', index=False)

    if args.protocol < 3:
        output_vars = statemonitors2dataframe([statemon_pre_neurons,
                                               statemon_post_neurons,
                                               ref_statemon_pre_neurons,
                                               ref_statemon_post_neurons])
        output_vars.to_csv(f'{args.save_path}/state_vars.csv', index=False)
    elif args.protocol == 3:
        output_spikes = pd.DataFrame(
            {'time_ms': np.array(active_monitor.t/defaultclock.dt),
             'num_fetch': np.array(active_monitor.event_count[0])})
        feather.write_dataframe(output_spikes, f'{args.save_path}/events_spikes.feather')
        # save weights to disk, directly from synapse object
        output_vars = pd.DataFrame(
            {'w_plast': np.hstack((stdp_synapse.w_plast,
                                   ref_stdp_synapse.w_plast)),
             'label': (['Proposed' for _ in range(len(stdp_synapse.w_plast))]
                      + ['Original' for _ in range(len(ref_stdp_synapse.w_plast))])
             })
        output_vars.to_csv(f'{args.save_path}/synapse_vars_weights.csv', index=False)

    if not args.quiet:
        if args.protocol == 1:
            delta_ti = 500
            delta_tf = 1000
            plt.scatter(spikemon_pre_neurons.t/ms, spikemon_pre_neurons.i)
            plt.scatter(spikemon_post_neurons.t/ms, spikemon_post_neurons.i)
            plt.build()
            plt.title('Spikes')
            plt.save_fig(f'{args.save_path}/fig1.txt', keep_colors=True)

            plt.clear_figure()
            plt.plot(ref_statemon_post_synapse.t[delta_ti:delta_tf]/ms,
                     ref_statemon_post_synapse.j_trace[0][delta_ti:delta_tf], label='base_post')
            plt.plot(ref_statemon_post_synapse.t[delta_ti:delta_tf]/ms,
                     ref_statemon_post_synapse.i_trace[0][delta_ti:delta_tf], label='base_pre')
            plt.title('Reference time window evolution')
            plt.build()
            plt.save_fig(f'{args.save_path}/fig2.txt', keep_colors=True)

            plt.clear_figure()
            plt.plot(statemon_post_neurons.t[delta_ti:delta_tf]/ms,
                     aux_plot_Ca(
                         statemon_post_neurons.Ca[0][delta_ti:delta_tf]), label='base_post')
            plt.plot(statemon_pre_neurons.t[delta_ti:delta_tf]/ms,
                     aux_plot_Ca(
                         statemon_pre_neurons.Ca[0][delta_ti:delta_tf]), label='base_pre')
            plt.title('Time window evolution')
            plt.build()
            plt.save_fig(f'{args.save_path}/fig3.txt', keep_colors=True)

            plt.clear_figure()
            plt.title('difference between Vms')
            plt.plot(ref_statemon_post_neurons.t/ms,
                     ref_statemon_post_neurons.Vm[0]/mV, label='ref')
            plt.plot(statemon_post_neurons.t/ms,
                     aux_plot_Vm(statemon_post_neurons.Vm[0]),
                     label='base')
            plt.build()
            plt.save_fig(f'{args.save_path}/fig4.txt', keep_colors=True)

            plt.clear_figure()
            plt.title('difference between weights')
            plt.plot(ref_statemon_post_synapse.t/ms,
                     ref_statemon_post_synapse.w_plast[0]/mV,
                     label='ref')
            plt.plot(statemon_post_synapse.t/ms,
                     aux_plot_Vm(statemon_post_synapse.w_plast[0]), label='base')
            plt.build()
            plt.save_fig(f'{args.save_path}/fig5.txt', keep_colors=True)

        elif args.protocol == 2:
            pairs_timing = (spikemon_post_neurons.t[:trial_duration]
                            - spikemon_post_neurons.t[:trial_duration][::-1])/ms
            plt.plot(pairs_timing,
                     aux_plot(statemon_post_synapse.w_plast[:, -1]))
            pairs_timing = (ref_spikemon_post_neurons.t[:trial_duration]
                            - ref_spikemon_post_neurons.t[:trial_duration][::-1])/ms
            plt.plot(pairs_timing, ref_statemon_post_synapse.w_plast[:, -1]/mV)
            plt.build()
            plt.save_fig(f'{args.save_path}/fig1.txt', keep_colors=True)

            plt.clear_figure()
            plt.plot(aux_plot_Ca(statemon_pre_neurons.Ca[0][:100]))
            plt.plot(aux_plot_Ca(statemon_post_neurons.Ca[0][:100]))
            plt.build()
            plt.save_fig(f'{args.save_path}/fig2.txt', keep_colors=True)

            plt.clear_figure()
            plt.plot(aux_plot_Ca(statemon_pre_neurons.Ca[29][:100]))
            plt.plot(aux_plot_Ca(statemon_post_neurons.Ca[29][:100]))
            plt.build()
            plt.save_fig(f'{args.save_path}/fig3.txt', keep_colors=True)

            plt.clear_figure()
            plt.scatter(spikemon_pre_neurons.t[:150]/ms, spikemon_pre_neurons.i[:150])
            plt.scatter(spikemon_post_neurons.t[:150]/ms, spikemon_post_neurons.i[:150])
            plt.build()
            plt.save_fig(f'{args.save_path}/fig4.txt', keep_colors=True)

            plt.clear_figure()
            plt.subplots(2, 1)
            plt.subplot(1, 1).title('Weight mean square error')
            plt.plot(((statemon_post_synapse.w_plast/mV - ref_statemon_post_synapse.w_plast/mV)**2).mean(axis=0), label='MSE')
            plt.subplot(2, 1).title('Vm mean square error')
            plt.plot(((statemon_post_neurons.Vm/mV - ref_statemon_post_neurons.Vm/mV)**2).mean(axis=0), label='MSE')
            plt.build()
            plt.save_fig(f'{args.save_path}/fig5.txt', keep_colors=True)

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
