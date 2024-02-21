from brian2 import defaultclock, ms, Hz, PoissonGroup, SpikeMonitor, StateMonitor,\
    device, prefs, run
import pandas as pd
import numpy as np
import json

from core.builder.groups_builder import create_synapses, create_neurons
from core.equations.neurons.sfp8LIF import sfp8LIF
from core.equations.synapses.sfp8CUBA import sfp8CUBA
from core.equations.synapses.sfp8iSTDP import sfp8iSTDP
from core.utils.misc import decimal2minifloat
from core.utils.prepare_models import set_hardwarelike_scheme,\
    generate_connection_indices
from core.utils.process_responses import statemonitors2dataframe


def istdp(args):
    defaultclock.dt = args.timestep * ms
    rng = np.random.default_rng()
    run_namespace = {}

    """ ================ models ================ """
    neuron_model = sfp8LIF()
    synapse_model = sfp8CUBA()
    istdp_model = sfp8iSTDP()
    
    if args.protocol == 1:
        tsim = 10001*ms
    elif args.protocol == 2:
        tsim = 30001*ms
        neuron_model.modify_model('namespace',
                                  decimal2minifloat(1),
                                  key='Ca_inc')
        istdp_model.modify_model('parameters',
                                 decimal2minifloat(-0.15625),
                                 key='target_rate')

    neuron_model.modify_model('events', args.event_condition, key='active_Ca',)
    # this results in around 5ms of refractory period
    neuron_model.modify_model('parameters', 20, key='alpha_refrac')
    neuron_model.modify_model('parameters', 55, key='alpha_syn')
    neuron_model.modify_model('parameters', 120, key='Iconst')

    num_exc = 8000
    num_inh = 2000
    num_input = 1000
    input_rate = 30*Hz
    poisson_pop = PoissonGroup(num_input, input_rate)
    neurons = create_neurons(num_exc+num_inh, neuron_model)
    exc_neurons = neurons[:num_exc]
    inh_neurons = neurons[num_exc:]

    """ ================ Wiring ================ """
    synapse_model.modify_model('connection', 0.03, key='p')
    input_synapse = create_synapses(poisson_pop,
                                    neurons,
                                    synapse_model)

    synapse_model.modify_model('connection', 0.02, key='p')
    # this is close to '2*1/N_incoming*mV'
    synapse_model.modify_model('parameters',
                               decimal2minifloat(0.013671875),
                               key='weight')
    exc_conn = create_synapses(exc_neurons, neurons, synapse_model)

    # this is close to '10*2*1/N_incoming*mV'
    synapse_model.modify_model('parameters',
                               decimal2minifloat(0.125),
                               key='weight')
    synapse_model.modify_model('namespace', 184, key='w_factor')
    inh_conn_static = create_synapses(inh_neurons, inh_neurons, synapse_model)

    sources, targets = generate_connection_indices(num_inh,
                                                   num_exc,
                                                   0.02)
    istdp_model.modify_model('connection', sources, key='i')
    istdp_model.modify_model('connection', targets, key='j')
    istdp_model.modify_model('parameters',
                             decimal2minifloat(0.125),
                             key='w_plast')
    istdp_synapse = create_synapses(inh_neurons,
                                    exc_neurons,
                                    istdp_model)

    set_hardwarelike_scheme(prefs, [neurons], defaultclock.dt, 'fp8')

    """ ================ Setting up monitors ================ """
    spikemon_exc_neurons = SpikeMonitor(exc_neurons,
                                        name='spikemon_exc_neurons')
    spikemon_inh_neurons = SpikeMonitor(inh_neurons,
                                         name='spikemon_inh_neurons')
    statemon_neurons = StateMonitor(exc_neurons,
                                    variables=['Ca'],
                                    record=True,
                                    dt=(tsim-1*ms)/200,
                                    name='statemon_neurons')
    statemon_synapses = StateMonitor(istdp_synapse,
                                     variables=['w_plast'],
                                     record=[x for x in range(len(sources))],
                                     dt=(tsim-1*ms)/2,
                                     name='statemon_synapses')

    metadata = {'event_condition': args.event_condition,
                'protocol': args.protocol,
                'duration': str(tsim)
                }
    with open(f'{args.save_path}/metadata.json', 'w') as f:
        json.dump(metadata, f)

    run(tsim, report='stdout', namespace=run_namespace)

    if args.backend == 'cpp_standalone' or args.backend == 'cuda_standalone':
        device.build(args.code_path)

    """ =================== Saving data =================== """
    output_spikes = pd.DataFrame(
        {'time_ms': np.array(spikemon_exc_neurons.t/defaultclock.dt),
         'id': np.array(spikemon_exc_neurons.i)}
    )
    output_spikes.to_csv(f'{args.save_path}/spikes_exc.csv', index=False)

    output_spikes = pd.DataFrame(
        {'time_ms': np.array(spikemon_inh_neurons.t/defaultclock.dt),
         'id': np.array(spikemon_inh_neurons.i)}
    )
    output_spikes.to_csv(f'{args.save_path}/spikes_inh.csv', index=False)

    state_vars = statemonitors2dataframe([statemon_synapses])
    state_vars.to_csv(f'{args.save_path}/weights.csv', index=False)
    state_vars = statemonitors2dataframe([statemon_neurons])
    state_vars.to_csv(f'{args.save_path}/state_vars.csv', index=False)
