""" This code is an adaptation of the ReScience publication made by
    Renan et al. (2017), see original (and complete) implementation in
    https://github.com/shimoura/ReScience-submission/tree/ShimouraR-KamijiNL-PenaRFO-CordeiroVL-CeballosCC-RomaroC-RoqueAC-2017/code/figures
    """

from brian2 import *


import pandas as pd
import numpy as np
import feather
import json

import gc

from core.equations.neurons.sfp8LIF import sfp8LIF
from core.equations.synapses.sfp8CUBA import sfp8CUBA
from core.equations.neurons.fp8LIF import fp8LIF
from core.equations.synapses.fp8CUBA import fp8CUBA
from core.builder.groups_builder import create_synapses, create_neurons
from core.utils.misc import decimal2minifloat
from core.utils.prepare_models import generate_connection_indices,\
    set_hardwarelike_scheme
from core.utils.process_responses import statemonitors2dataframe

def fp8_potjans_diesmann(args):
    if args.protocol == 1:
        tsim = 60
    elif args.protocol == 2:
        tsim = 100
    else:
        raise UserWarning('A protocol must be provided.')

    defaultclock.dt = args.timestep * ms

    rng = np.random.default_rng()

    """ =================== Parameters =================== """
    ###########################################################################
    # Population size per layer
    #          2/3e   2/3i   4e    4i    5e    5i    6e     6i    Th
    n_layer = [20683, 5834, 21915, 5479, 4850, 1065, 14395, 2948, 902]

    # Total cortical Population
    N = sum(n_layer[:-1])

    # Number of neurons accumulated
    nn_cum = [0]
    nn_cum.extend(cumsum(n_layer))

    # Prob. connection table: from colum to row
    #                L2/3e 	L2/3i 	L4e 	L4i   L5e   L5i 	L6e    L6i 	   Th
    table = array([[0.101,  0.169, 0.044, 0.082, 0.032, 0.,     0.008, 0.,     0.    ],  #L2/3e
                   [0.135,  0.137, 0.032, 0.052, 0.075, 0.,     0.004, 0.,     0.    ],  #L2/3i
                   [0.008,  0.006, 0.050, 0.135, 0.007, 0.0003, 0.045, 0.,     0.0983],  #L4e
                   [0.069,  0.003, 0.079, 0.160, 0.003, 0.,     0.106, 0.,     0.0619],  #L4i
                   [0.100,  0.062, 0.051, 0.006, 0.083, 0.373,  0.020, 0.,     0.    ],  #L5e
                   [0.055,  0.027, 0.026, 0.002, 0.060, 0.316,  0.009, 0.,     0.    ],  #L5i
                   [0.016,  0.007, 0.021, 0.017, 0.057, 0.020,  0.040, 0.225,  0.0512],  #L6e
                   [0.036,  0.001, 0.003, 0.001, 0.028, 0.008,  0.066, 0.144,  0.0196]]) #L6i

    w_ex = 1.25
    w_ex_2 = 2.5
    delay_ex = 1.5
    delay_in = .8

    """ =============== Neuron definitions =============== """
    if args.rounding == 'stochastic':
        neu_model = sfp8LIF()
    else:
        neu_model = fp8LIF()
    neu_model.modify_model('parameters', '43', key='alpha_syn')
    neurons = create_neurons(N, neu_model)
    sampled_var = np.clip(rng.normal(448, 10, N), 0, 480)
    neurons.Vm = decimal2minifloat(sampled_var, raise_warning=False)

    """ ==================== Networks ==================== """
    bg_layer = array([1600, 1500 ,2100, 1900, 2000, 1900, 2900, 2100])

    # Stores NeuronGroups, one for each population
    pop = []
    for r in range(0, 8):
        pop.append(neurons[nn_cum[r]:nn_cum[r+1]])

    # Stores connections
    con = []
    pre_index = []
    post_index = []

    for c in range(0, 8):
        for r in range(0, 8):
            # number of synapses calculated with equation 3 from the article
            nsyn = int(log(1.0-table[r][c])/log(1.0 - (1.0/float(n_layer[c]*n_layer[r]))))

            pre_index = randint(n_layer[c], size=nsyn)
            post_index = randint(n_layer[r], size=nsyn)

            if nsyn < 1:
                pass
            else:
                if args.rounding == 'stochastic':
                    syn_model = sfp8CUBA()
                else:
                    syn_model = fp8CUBA()
                syn_model.modify_model('connection', pre_index, key='i')
                syn_model.modify_model('connection', post_index, key='j')
                con.append(create_synapses(pop[c], pop[r], syn_model))
                # Excitatory connections
                if (c % 2) == 0:
                    # Synaptic weight from L4e to L2/3e is doubled
                    if c == 2 and r == 0:
                        sampled_var = np.clip(rng.normal(w_ex_2,
                                                         w_ex_2/10,
                                                         nsyn),
                                              0,
                                              480)
                        con[-1].weight = decimal2minifloat(sampled_var,
                                                           raise_warning=False)
                    else:
                        sampled_var = np.clip(rng.normal(w_ex,
                                                         w_ex/10,
                                                         nsyn),
                                              0,
                                              480)
                        con[-1].weight = decimal2minifloat(sampled_var,
                                                           raise_warning=False)

                    sampled_var = np.clip(rng.normal(delay_ex,
                                                     delay_ex/2,
                                                     nsyn),
                                          1, np.inf)
                    con[-1].delay = np.rint(sampled_var)*ms

                # Inhibitory connections
                else:
                    sampled_var = np.clip(rng.normal(args.w_in,
                                                     args.w_in/10,
                                                     nsyn),
                                          0,
                                          480)
                    con[-1].weight = decimal2minifloat(sampled_var,
                                                       raise_warning=False)
                    con[-1].namespace['w_factor'] = 184  # -1 in decimal

                    sampled_var = np.clip(rng.normal(delay_in,
                                                     delay_in/2,
                                                     nsyn),
                                          1, np.inf)
                    con[-1].delay = np.rint(sampled_var)*ms

    bg_in = []
    poisson_pop = []
    if args.rounding == 'stochastic':
        syn_model = sfp8CUBA()
    else:
        syn_model = fp8CUBA()
    syn_model.connection['p'] = .03
    for r in range(0, 8):
        poisson_pop.append(PoissonGroup(bg_layer[r], rates=args.bg_freq*Hz))
        bg_in.append(create_synapses(poisson_pop[-1], pop[r], syn_model))

    if args.protocol == 2:
        thal_con = []
        thal_input = []
        # More rate was used (orginal was 120) to elicit comparable activity
        stimulus = TimedArray(np.tile([0 for _ in range(70)]
                                      + [240]
                                      + [0 for _ in range(29)], tsim)*Hz,
                       dt=10.*ms)
        thal_input = PoissonGroup(n_layer[8], rates='stimulus(t)')
        thal_nsyn = []
        for r in range(0, 8):
            sources, targets = generate_connection_indices(thal_input.N,
                                                           pop[r].N,
                                                           table[r][8])
            if not np.any(sources):
                continue
            thal_nsyn.append(len(sources))
            syn_model.modify_model('connection', sources, key='i')
            syn_model.modify_model('connection', targets, key='j')
            thal_con.append(create_synapses(thal_input, pop[r], syn_model))

    # Required to emulate hardware
    set_hardwarelike_scheme(prefs, [neurons], defaultclock.dt,
                            'fp8')

    ###########################################################################
    # Creating spike monitors
    ###########################################################################
    smon_net = SpikeMonitor(neurons)
    mon_vars = StateMonitor(neurons,
                            variables=['Vm', 'Ca', 'g'],
                            record=[nn_cum[x] for x in range(8)],
                            name='neu_state_variables')

    """ ==================== Running ===================== """
    net = Network(collect())

    net.add(neurons, pop, con, bg_in, poisson_pop)

    if args.protocol == 1:
        net.run(tsim*second, report='stdout')
        if args.backend == 'cpp_standalone': device.build(args.code_path)
    elif args.protocol == 2:
        net.add(thal_input, thal_con)

        for i, nsyn in enumerate(thal_nsyn):
            sampled_var = np.clip(rng.normal(10*w_ex_2, 10*w_ex_2/10, nsyn),
                                  0,
                                  480)
            thal_con[i].weight = decimal2minifloat(sampled_var,
                                                   raise_warning=False)
        net.run(tsim*second, report='stdout')
        if args.backend == 'cpp_standalone': device.build(args.code_path)
    gc.collect()

    """ ==================== Plotting ==================== """
    data = pd.DataFrame({'i': np.array(smon_net.i),
                         't': np.array(smon_net.t/defaultclock.dt)})

    # cortical layer labels: e for excitatory; i for inhibitory
    lname = ['L23e', 'L23i', 'L4e', 'L4i', 'L5e', 'L5i', 'L6e', 'L6i']

    # number of neurons by layer
    n_layer = [0] + n_layer[:-1]
    l_bins = np.cumsum(n_layer)  # cumulative number of neurons by layer

    # grouping spiking times for each neuron
    keys, values = data.sort_values(['i', 't']).values.T
    ukeys, index = np.unique(keys, True)
    arrays = np.split(values, index[1:])

    spk_neuron = pd.DataFrame({'i': range(0, N), 't': [[]]*N})
    spk_neuron.iloc[ukeys.astype(int), 1] = arrays

    # creating a flag to identify cortical layers
    spk_neuron['layer'] = pd.cut(spk_neuron['i'],
                                 l_bins,
                                 labels=lname,
                                 right=False)
    feather.write_dataframe(spk_neuron, args.save_path + 'spikes.feather')

    output_vars = statemonitors2dataframe([mon_vars])
    feather.write_dataframe(output_vars, f'{args.save_path}/synapse_vars.feather')

    Metadata = {'dt': str(defaultclock.dt),
                'duration': str(tsim*second),
                'mean_inh_w': str(args.w_in),
                'protocol': args.protocol,
                'background_rate': str(args.bg_freq)}
    with open(args.save_path + 'metadata.json', 'w') as f:
        json.dump(Metadata, f)
