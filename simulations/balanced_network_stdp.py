import numpy as np
import feather
import pandas as pd
import json
import quantities as q
import sys
import gc

from brian2 import run, set_device, device, defaultclock, scheduling_summary,\
    SpikeGeneratorGroup, StateMonitor, SpikeMonitor, TimedArray, PoissonInput,\
    prefs, scheduling_summary, PoissonGroup
from brian2 import mV, second, Hz, ms, nS, pF

import plotext as plt

from core.utils.misc import minifloat2decimal, decimal2minifloat
from core.utils.process_responses import neurons_rate, statemonitors2dataframe,\
    objects2dataframe
from core.utils.prepare_models import generate_connection_indices,\
    set_hardwarelike_scheme

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


def balanced_network_stdp(args):
    defaultclock.dt = args.timestep * ms
    rng = np.random.default_rng()
    run_namespace = {}

    """ ================ models and helper functions ================ """
    if args.precision == 'fp8':
        neuron_model = sfp8LIF()
        def aux_w_sample(x): return decimal2minifloat(x)
        def aux_plot(x): return minifloat2decimal(x)
        def aux_plot_Ca(x): return minifloat2decimal(x)
        synapse_model = sfp8CUBA()
        stdp_model = sfp8STDP()
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
    tmax = args.tsim * second
    Ne, Ni = 90000, 22500
    Nt = Ne + Ni
    exc_delay = 1.5
    ext_weights = 1 if args.precision == 'fp64' else 10
    exc_weights = args.we
    inh_weights = 5 * exc_weights
    conn_condition = 'i!=j'
    p_conn = .1

    if args.precision == 'fp8':
        neuron_model.modify_model('parameters', 'ceil(127*rand())', key='Vm')
        neuron_model.modify_model('parameters', args.ca_decays, key='alpha_ca')
        neuron_model.modify_model('parameters', '45', key='alpha_syn')
        neuron_model.modify_model('parameters', '54', key='alpha')
    elif args.precision == 'fp64':
        neuron_model.modify_model('parameters',
                                  'clip(5.7 + randn()*7.2, 0, inf)*mV',
                                  key='Vm')
        neuron_model.modify_model('parameters', args.ca_decays, key='tau_ca')
        neuron_model.modify_model('refractory', '1*ms')
        neuron_model.modify_model('parameters', '.66*ms', key='tau_syn')
        neuron_model.modify_model('namespace', 25*nS, key='gl')
        neuron_model.modify_model('namespace', 250*pF, key='Cm')
    neuron_model.modify_model('events', args.event_condition, key='active_Ca')

    neurons = create_neurons(Ne+Ni,
                             neuron_model)
    exc_neurons = neurons[:Ne]
    inh_neurons = neurons[Ne:]
    # Specific neurons to make recording easier
    mon_neurons = create_neurons(2, neuron_model)
    neuron_model.model += 'event_count : 1\n'
    mon_active_neurons = create_neurons(1, neuron_model)
    mon_active_neurons.run_regularly(
        'event_count = 0',
        name=f'clear_event_counter',
        dt=1*ms,
        when='after_synapses',
        order=1)

    if args.precision == 'fp8':
        poisson_spikes_1 = PoissonGroup(9000, 2.32*Hz)
        synapse_model.modify_model('connection', 0.1, key='p')
        synapse_model.modify_model('parameters',
                                   decimal2minifloat(ext_weights),
                                   key='weight')
        input_synapse_1 = create_synapses(poisson_spikes_1,
                                          neurons,
                                          synapse_model)
        input_synapse_2 = create_synapses(poisson_spikes_1,
                                          mon_neurons,
                                          synapse_model)
    elif args.precision == 'fp64':
        poisson_spikes_1 = PoissonInput(neurons, 'g', 9000, rate=2.32*Hz,
                                        weight=f'{ext_weights}*mV')
        poisson_spikes_2 = PoissonInput(mon_neurons, 'g', 9000, rate=2.32*Hz,
                                        weight=f'{ext_weights}*mV')

    tmp_eq = f'int(clip({exc_delay} + randn()*{exc_delay/2}, 1, inf))*ms'
    stdp_model.modify_model('parameters', tmp_eq, key='delay')
    synapse_model.modify_model('parameters', tmp_eq, key='delay')

    stdp_model.modify_model('connection', p_conn, key='p')
    synapse_model.modify_model('connection', p_conn, key='p')

    stdp_model.modify_model('connection', conn_condition, key='condition')
    synapse_model.modify_model('connection', conn_condition, key='condition')

    w_max = args.w_max
    fp8_w_max = decimal2minifloat(w_max)

    if args.precision == 'fp8':
        stdp_model.modify_model('parameters',
                                f'ceil({fp8_w_max}*rand())',
                                key='w_plast')
        synapse_model.modify_model('parameters',
                                   decimal2minifloat(exc_weights),
                                   key='weight')
        # TODO remove
        #stdp_model.modify_model('namespace', 1, key='eta')
    elif args.precision == 'fp64':
        stdp_model.modify_model('parameters', exc_weights*mV, key='w_plast')
        synapse_model.modify_model('parameters', exc_weights*mV, key='weight')
        stdp_model.modify_model('namespace', 0.1*mV, key='eta')

    turnoff_t = 195000
    alpha = args.alpha
    if args.precision == 'fp8':
        stdp_model.modify_model('on_pre',
            stdp_model.on_pre['stdp_fanout'] + f'\nw_plast = int(w_plast>{fp8_w_max})*{fp8_w_max} + int(w_plast<={fp8_w_max})*w_plast\n', 
            key='stdp_fanout')
    elif args.precision == 'fp64':
        if args.protocol == 1:
            plasticity_rule = f'''
                delta_w = int(Ca_pre>0 and Ca_post<0)*(eta*Ca_pre) - int(Ca_pre<0 and Ca_post>0)*(eta*Ca_post)
                w_plast = clip(w_plast + delta_w*int(t<{turnoff_t}*ms), 0*mV, {w_max}*mV)'''
        elif args.protocol == 2:
            plasticity_rule = f'''
                delta_w = int(Ca_pre>0 and Ca_post<0)*(eta*Ca_pre*(0.04**0.6)*(w_plast/mV)**0.4) - int(Ca_pre<0 and Ca_post>0)*(eta*Ca_post*{alpha}*(w_plast/mV))
                w_plast = clip(w_plast + delta_w*int(t<{turnoff_t}*ms), 0*mV, {w_max}*mV)'''
        stdp_model.on_pre['stdp_fanout'] = plasticity_rule

    stdp_synapse = create_synapses(exc_neurons,
                                   exc_neurons,
                                   stdp_model)
    exc_synapse = create_synapses(exc_neurons,
                                  inh_neurons,
                                  synapse_model)
    if args.precision == 'fp8':
        tmp_inh_weights = decimal2minifloat(inh_weights)
        tmp_eq = f'ceil(randn() + {tmp_inh_weights})'
        synapse_model.modify_model('parameters', tmp_eq, key='weight')
        synapse_model.modify_model('namespace', 184, key='w_factor')
    elif args.precision == 'fp64':
        tmp_eq = f'clip({inh_weights} + randn()*{inh_weights/10}, 0, inf)*mV'
        synapse_model.modify_model('parameters', tmp_eq, key='weight')
        synapse_model.modify_model('namespace', -1, key='w_factor')
    inh_synapse = create_synapses(inh_neurons,
                                  neurons,
                                  synapse_model)

    stdp_mon_neuron_1 = create_synapses(exc_neurons, mon_neurons, stdp_model)
    stdp_mon_neuron_2 = create_synapses(inh_neurons, mon_neurons, synapse_model)
    stdp_mon_neuron_3 = create_synapses(mon_neurons, neurons, stdp_model)

    stdp_model.on_pre['stdp_fanout'] = f'''
        event_count_post += 1
        '''
    stdp_model.modify_model('connection', 1., key='p')
    stdp_mon_events = create_synapses(exc_neurons, mon_active_neurons, stdp_model)

    # Required to emulate hardware
    set_hardwarelike_scheme(prefs, [neurons, mon_neurons], defaultclock.dt,
                            args.precision)

    """ ================ Setting up monitors ================ """
    # For longer simulations, much higher dt might be necessary
    spikemon_neurons = SpikeMonitor(neurons,
                                    name='spikemon_neurons')
    stdpmon_incoming = StateMonitor(stdp_mon_neuron_1,
                                    variables=['w_plast'],
                                    record=[0, 1],
                                    #dt=1000*ms,
                                    dt=5*ms,
                                    name='stdp_in_w')
    stdpmon_outgoing = StateMonitor(stdp_mon_neuron_3,
                                    variables=['w_plast'],
                                    record=[0, 1],
                                    #dt=1000*ms,
                                    dt=5*ms,
                                    name='stdp_out_w')
    mon_neurons_vars = StateMonitor(mon_neurons,
                                    variables=['Vm', 'Ca', 'g'],
                                    record=[0, 1],
                                    #dt=100*ms,
                                    dt=5*ms,
                                    name='neu_state_variables')
    active_monitor = StateMonitor(mon_active_neurons,
                                  'event_count',
                                  record=0,
                                  when='after_synapses',
                                  order=0,
                                  name='event_count_monitor')

    metadata = {'dt': str(defaultclock.dt),
                'duration': str(tmax),
                'precision': args.precision,
                'eta': stdp_model.namespace['eta'],
                'N_exc': Ne,
                'N_inh': Ni,
                'w_max': str(w_max*mV),
                'turnoff_t': str(turnoff_t*ms),
                'alpha': alpha,
                'inh_weights': str(inh_weights*mV),
                'protocol': args.protocol,
                'event_condition': args.event_condition,
                'init_we': args.we,
                'ca_decays': args.ca_decays
                }
    with open(f'{args.save_path}/metadata.json', 'w') as f:
        json.dump(metadata, f)

    run(tmax, report='stdout', namespace=run_namespace)
    gc.collect()

    if args.backend == 'cpp_standalone' or args.backend == 'cuda_standalone':
        device.build(args.code_path)

    output_spikes = pd.DataFrame(
        {'time_ms': np.array(spikemon_neurons.t/defaultclock.dt),
         'id': np.array(spikemon_neurons.i)})
    output_spikes.to_csv(f'{args.save_path}/output_spikes.csv', index=False)

    output_vars = statemonitors2dataframe([mon_neurons_vars,
                                           stdpmon_incoming,
                                           stdpmon_outgoing])
    output_vars.to_csv(f'{args.save_path}/output_vars.csv', index=False)
    obj_vars = objects2dataframe([stdp_synapse], [('i', 'j', 'w_plast')])
    feather.write_dataframe(obj_vars, f'{args.save_path}/obj_vars.feather')

    output_spikes = pd.DataFrame(
        {'time_ms': np.array(active_monitor.t/defaultclock.dt),
         'num_events': np.array(active_monitor.event_count[0])})
    output_spikes.to_csv(f'{args.save_path}/events_spikes.csv', index=False)

    if not args.quiet:
        print(scheduling_summary())
        print("Plastic synapses:")
        print(pd.DataFrame(np.array(stdp_synapse.w_plast)).describe())
        print("Inhibitory synapses:")
        print(pd.DataFrame(np.array(inh_synapse.weight)).describe())
        print("Excitatory synapses:")
        print(pd.DataFrame(np.array(exc_synapse.weight)).describe())
