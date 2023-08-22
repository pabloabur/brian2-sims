import numpy as np
import feather
import pandas as pd
import json
import quantities as q
import sys
import gc

from brian2 import run, set_device, device, defaultclock, scheduling_summary,\
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


def balanced_network_stdp(args):
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

    """ ================ Simulation specifications ================ """
    tmax = 200 * second
    Ne, Ni = 90000, 22500
    Nt = Ne + Ni
    exc_delay = 1.5
    ext_weights = .25
    exc_weights = 4.561
    inh_weights = 5 * exc_weights
    conn_condition = 'i!=j'
    p_conn = .1

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

    # TODO this is just a test, organize it
    stdp_model.on_pre['stdp_fanout'] = '''
        delta_w = int(Ca_pre>0 and Ca_post<0)*(eta*Ca_pre) - int(Ca_pre<0 and Ca_post>0)*(eta*Ca_post)
        w_plast = clip(w_plast + delta_w*int(t<140000*ms), 0*mV, 100*mV)'''

    # TODO choose one delta w
    #delta_w = int(Ca_pre>0 and Ca_post<0)*(eta*Ca_pre*(w_plast/mV)**0.4) - int(Ca_pre<0 and Ca_post>0)*(eta*Ca_post*(w_plast/mV))

    stdp_synapse = create_synapses(exc_neurons,
                                   neurons,
                                   stdp_model)
    inh_synapse = create_synapses(inh_neurons,
                                  neurons,
                                  synapse_model)
    stdp_mon_neuron_1 = create_synapses(exc_neurons, mon_neurons, stdp_model)
    stdp_mon_neuron_2 = create_synapses(inh_neurons, mon_neurons, synapse_model)
    stdp_mon_neuron_3 = create_synapses(mon_neurons, neurons, stdp_model)

    stdp_synapse.namespace['eta'] = 0.001*mV

    # Required to emulate hardware
    set_hardwarelike_scheme(prefs, [neurons, mon_neurons], defaultclock.dt)

    """ ================ Setting up monitors ================ """
    spikemon_neurons = SpikeMonitor(neurons,
                                    name='spikemon_neurons')
    if args.protocol == 1:
        stdpmon_incoming = StateMonitor(stdp_mon_neuron_1,
                                        variables=['w_plast'],
                                        record=[0, 1],
                                        dt=1000*ms,
                                        name='stdp_in_w')
        stdpmon_outgoing = StateMonitor(stdp_mon_neuron_3,
                                        variables=['w_plast'],
                                        record=[0, 1],
                                        dt=1000*ms,
                                        name='stdp_out_w')
        mon_neurons_vars = StateMonitor(mon_neurons,
                                        variables=['Ca', 'g'],
                                        record=[0, 1],
                                        dt=100*ms,
                                        name='neu_state_variables')
    elif args.protocol == 2:
        spikemon_neurons_test = SpikeMonitor(mon_neurons)
        active_monitor = EventMonitor(neurons, 'active_Ca')

    run(tmax, report='stdout', namespace=run_namespace)
    gc.collect()

    if args.backend == 'cpp_standalone' or args.backend == 'cuda_standalone':
        device.build(args.code_path)

    metadata = {'dt': str(defaultclock.dt),
                'duration': str(tmax),
                'N_exc': Ne,
                'N_inh': Ni}
    with open(f'{args.save_path}/metadata.json', 'w') as f:
        json.dump(metadata, f)

    if not args.quiet:
        # TODO more stuff in 1, and 2 was not even worked out
        # TODO this is not quiet
        if args.protocol == 1:
            output_spikes = pd.DataFrame(
                {'time_ms': np.array(spikemon_neurons.t/defaultclock.dt),
                 'id': np.array(spikemon_neurons.i)})
            output_spikes.to_csv(f'{args.save_path}/output_spikes.csv', index=False)

            output_vars = statemonitors2dataframe([mon_neurons_vars,
                                                   stdpmon_incoming,
                                                   stdpmon_outgoing])
            output_vars.to_csv(f'{args.save_path}/output_vars.csv', index=False)

        if args.protocol == 2:
            num_fetches = {'pre': spikemon_neurons.num_spikes,
                           'fanout': active_monitor.num_events}
            print(f'Potential memory fetches for each strategy:\nConventional:'
                  f' {2*num_fetches["pre"]/1e6}M\nFanout: '
                  f'{num_fetches["fanout"]/1e6}M')
            print('Number of plastic connections')
            print(np.shape(stdp_synapse.w_plast))
            # TODO remove, they're the same
            print('spkmon numspike')
            print(spikemon_neurons.num_spikes)
            print('spkmon numevents')
            print(active_monitor.num_events)
            print('shape of spk t of each above')
            print(np.shape(spikemon_neurons.t))
            print(np.shape(active_monitor.t))

        max_synapse_id = np.where(stdp_synapse.w_plast/mV==np.max(stdp_synapse.w_plast/mV))[0]
        target_neuron_id = stdp_synapse.j[max_synapse_id[0]]

        # target neuron has at least one saturated weight
        plt.clear_figure()
        plt.hist(np.array(stdp_synapse.w_plast/mV)[stdp_synapse.j==target_neuron_id])
        plt.title(f'Incoming weights to neuron {int(target_neuron_id)}')
        plt.build()
        plt.save_fig(f'{args.save_path}/fig1.txt', keep_colors=True)

        plt.clear_figure()
        plt.hist(np.array(stdp_synapse.w_plast/mV)[stdp_synapse.i==target_neuron_id])
        plt.title(f'Outgoing weights from neuron {int(target_neuron_id)}')
        plt.build()
        plt.save_fig(f'{args.save_path}/fig12.txt', keep_colors=True)

        random_id = 100
        plt.clear_figure()
        plt.hist(np.array(stdp_synapse.w_plast/mV)[stdp_synapse.j==random_id])
        plt.title(f'Weights targetting neuron {int(random_id)}')
        plt.build()
        plt.save_fig(f'{args.save_path}/fig13.txt', keep_colors=True)

        plt.clear_figure()
        plt.hist(np.array(stdp_synapse.w_plast/mV)[stdp_synapse.i==random_id])
        plt.title(f'Weights from neuron {int(random_id)}')
        plt.build()
        plt.save_fig(f'{args.save_path}/fig14.txt', keep_colors=True)

        plt.clear_figure()
        plt.hist(stdp_synapse.w_plast/mV)
        plt.title('Final distribution of weights')
        plt.build()
        plt.save_fig(f'{args.save_path}/fig2.txt', keep_colors=True)

        #neu_r = neurons_rate(spikemon_neurons_test, tmax/ms)
        #plt.clear_figure()
        #plt.plot(neu_r.times/q.ms, neu_r[:, 0].magnitude.flatten())
        #plt.plot(neu_r.times/q.ms, neu_r[:, 1].magnitude.flatten())
        #plt.title('Rate of some neurons')
        #plt.build()
        #plt.save_fig(f'{args.save_path}/fig3.txt', keep_colors=True)
