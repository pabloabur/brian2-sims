import numpy as np
import feather
import pandas as pd
import quantities as q
import sys
import gc

from brian2 import run, set_device, device, defaultclock, scheduling_summary,\
    SpikeGeneratorGroup, StateMonitor, SpikeMonitor, TimedArray, PoissonInput,\
    prefs, EventMonitor
from brian2 import mV, second, Hz, ms

import plotext as plt

from core.utils.misc import minifloat2decimal, decimal2minifloat
from core.utils.process_responses import neurons_rate
from core.utils.prepare_models import generate_connection_indices

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


def balanced_network_stdp(args):
    defaultclock.dt = args.timestep * ms
    prefs.core.network.default_schedule = ['start', 'groups', 'thresholds',
                                           'resets', 'synapses', 'end']
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

    """ ================ Simulation specifications ================ """
    tmax = 60000 * ms
    Ne, Ni = 54000, 13500
    Nt = Ne + Ni
    exc_delay = 1.5
    ext_weights = .25
    exc_weights = 4.561
    inh_weights = 5 * exc_weights
    conn_condition = 'i!=j'
    p_conn = .00001 # TODO was .1

    neuron_model.modify_model('parameters',
                              'clip(5.7 + randn()*7.2, 0, inf)*mV',
                              key='Vm')

    neurons = create_neurons(Ne+Ni,
                             neuron_model)
    exc_neurons = neurons[:Ne]
    inh_neurons = neurons[Ne:]
    # Specific neurons to make recording easier
    mon_neurons = create_neurons(2, neuron_model)

    poisson_spikes_1 = PoissonInput(neurons, 'g', 9000, rate=2.32*Hz,
                                    weight=f'{ext_weights}*mV')
    poisson_spikes_2 = PoissonInput(mon_neurons, 'g', 9000, rate=2.32*Hz,
                                    weight=f'{ext_weights}*mV')

    tmp_eq = f'int(clip({exc_delay} + randn()*{exc_delay/2}, 1, inf))*ms'
    stdp_model.modify_model('parameters', tmp_eq, key='delay')
    synapse_model.modify_model('parameters', tmp_eq, key='delay')

    stdp_model.modify_model('connection', p_conn, key='p')
    synapse_model.modify_model('connection', p_conn, key='p')

    tmp_eq = f'clip({inh_weights} + randn()*{inh_weights/10}, 0, inf)*mV'
    synapse_model.modify_model('parameters', tmp_eq, key='weight')
    synapse_model.modify_model('namespace', -1, key='w_factor')
    tmp_eq = f'clip({exc_weights} + randn()*{exc_weights/10}, 0, inf)*mV'
    stdp_model.modify_model('parameters', tmp_eq, key='w_plast')

    stdp_model.modify_model('connection', conn_condition, key='condition')
    synapse_model.modify_model('connection', conn_condition, key='condition')

    # TODO choose one delta w
    #delta_w = int(Ca_pre>0 and Ca_post<0)*(eta*Ca_pre*(w_plast/mV)**0.4) - int(Ca_pre<0 and Ca_post>0)*(eta*Ca_post*(w_plast/mV))

    stdp_synapse = create_synapses(exc_neurons,
                                   neurons,
                                   stdp_model)
    inh_synapse = create_synapses(inh_neurons,
                                  neurons,
                                  synapse_model)
    mon_stdp_1 = create_synapses(exc_neurons, mon_neurons, stdp_model)
    mon_stdp_2 = create_synapses(inh_neurons, mon_neurons, synapse_model)
    mon_stdp_3 = create_synapses(mon_neurons, neurons, stdp_model)

    # TODO this should be a module/function,  e.g. add_tsv_scheme(), regardless of precision,
    # where run_reg options are sent and a flag is used when groups are being created
    stdp_synapse.namespace['eta'] = 2**-9*mV
    if args.precision == 'fp64':
       neurons.run_regularly(
           'Ca = Ca*int(Ca>0) - Ca*int(Ca<0)',
           name='clear_spike_flag_post',
           dt=defaultclock.dt,
           when='after_synapses',
           order=1)

    """ ================ Setting up monitors ================ """
    spikemon_neurons = SpikeMonitor(neurons,
                                    name='spikemon_neurons')
    # TODO this is not a neuron; maybe connections here
    # TODO might just calculate avg inc conn per neuron? what for...
    # TODO dont think i'll be able to plot that many better just look at rate and final w
    # TODO ca is alsoo good, could also clip it too
    # TODO maybe average weights and show it overtime?
    stdpmon_incoming = StateMonitor(mon_stdp_1,
                                    variables=['w_plast'],
                                    record=[0, 1],
                                    dt=50*ms)
    stdpmon_outgoing = StateMonitor(mon_stdp_3,
                                    variables=['w_plast'],
                                    record=[0, 1],
                                    dt=50*ms)
    #statemon_neurons = StateMonitor(neurons,
    #                                     variables=['Vm', 'g', 'Ca'],
    #                                     record=range(N_post),
    #                                     name='statemon_neurons')
    active_monitor = EventMonitor(neurons, 'active_Ca')

    run(tmax, report='stdout', namespace=run_namespace)
    gc.collect()

    import pdb;pdb.set_trace()
    if args.backend == 'cpp_standalone':
        device.build(args.code_path) # TODO clean=True?

    if not args.quiet:
        num_fetches = {'pre': spikemon_neurons.num_spikes,
                       'fanout': active_monitor.num_events}
        print(num_fetches)
        print(f'Potential memory fetches for each strategy:\nConventional:'
              f' {2*num_fetches["pre"]/1e6}M\nFanout: '
              f'{num_fetches["fanout"]/1e6}M')
        print('shape of wplast')
        print(np.shape(stdp_synapse.w_plast))
        print('shape of spkmon numspike')
        print(np.shape(spikemon_neurons.num_spikes))
        print('shape of spkmon numevents')
        print(np.shape(active_monitor.num_events))
        print('shape of spk t of each above')
        print(np.shape(spikemon_neurons.t))
        print(np.shape(active_monitor.t))

        max_weight_idx = np.where(stdp_synapse.w_plast/mV==np.max(stdp_synapse.w_plast/mV))[0]
        target_id = stdp_synapse.j[max_weight_idx[0]]
        #source_ids = np.array(stdp_synapse.i)[stdp_synapse.j==target_id]
        #n_incoming = np.shape(statemon_post_synapse.w_plast[stdp_synapse.j==target_id, :])[0]

        plt.clear_figure()
        plt.hist(np.array(stdp_synapse.w_plast/mV)[stdp_synapse.j==target_id])
        plt.title('Weights targetting a neuron')
        plt.build()
        plt.save_fig(f'{args.save_path}/fig1.txt', keep_colors=True)

        plt.clear_figure()
        plt.hist(stdp_synapse.w_plast/mV)
        plt.title('Final distribution of weights')
        plt.build()
        plt.save_fig(f'{args.save_path}/fig2.txt', keep_colors=True)

        neu_r = neurons_rate(spikemon_neurons, tmax/ms)
        plt.clear_figure()
        plt.plot(neu_r.times/q.ms, neu_r[:, target_id].magnitude.flatten())
        plt.plot(neu_r.times/q.ms, neu_r[:, source_ids[0]].magnitude.flatten())
        plt.title('Rate of some neurons')
        plt.build()
        plt.save_fig(f'{args.save_path}/fig3.txt', keep_colors=True)
