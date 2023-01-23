""" This code is an adaptation of the ReScience publication made by
    Renan et al. (2017), see original (and complete) implementation in
    https://github.com/shimoura/ReScience-submission/tree/ShimouraR-KamijiNL-PenaRFO-CordeiroVL-CeballosCC-RomaroC-RoqueAC-2017/code/figures
    """

from brian2 import *


import pandas as pd
import numpy as np
import scipy.stats as sc
import feather
import json

import gc
from datetime import datetime

from core.equations.neurons.int8LIF import int8LIF
from core.equations.synapses.int8CUBA import int8CUBA
from core.builder.groups_builder import create_synapses, create_neurons
from core.utils.misc import minifloat2decimal, decimal2minifloat
from core.utils.prepare_models import generate_connection_indices


def int8_potjans_diesmann(args):
    if args.protocol == 1:
        tsim = 2
    elif args.protocol == 2:
        tsim = 100
    else:
        raise UserWarning('A protocol must be provided.')

    defaultclock.dt = args.timestep * ms

    rng = np.random.default_rng()

    """ =================== Parameters =================== """
    ###############################################################################
    # Population size per layer
    #          2/3e   2/3i   4e    4i    5e    5i    6e     6i    Th
    n_layer = [20683, 5834, 21915, 5479, 4850, 1065, 14395, 2948, 902]

    # Total cortical Population
    N = sum(n_layer[:-1])

    # Number of neurons accumulated
    nn_cum = [0]
    nn_cum.extend(cumsum(n_layer))

    # Prob. connection table
    table = array([[0.101,  0.169, 0.044, 0.082, 0.032, 0.,     0.008, 0.,     0.    ],
                   [0.135,  0.137, 0.032, 0.052, 0.075, 0.,     0.004, 0.,     0.    ],
                   [0.008,  0.006, 0.050, 0.135, 0.007, 0.0003, 0.045, 0.,     0.0983],
                   [0.069,  0.003, 0.079, 0.160, 0.003, 0.,     0.106, 0.,     0.0619],
                   [0.100,  0.062, 0.051, 0.006, 0.083, 0.373,  0.020, 0.,     0.    ],
                   [0.055,  0.027, 0.026, 0.002, 0.060, 0.316,  0.009, 0.,     0.    ],
                   [0.016,  0.007, 0.021, 0.017, 0.057, 0.020,  0.040, 0.225,  0.0512],
                   [0.036,  0.001, 0.003, 0.001, 0.028, 0.008,  0.066, 0.144,  0.0196]])

    w_ex = 2
    w_ex_2 = 4
    delay_ex = 1.5
    delay_in = .8

    """ =============== Neuron definitions =============== """
    neu_model = int8LIF()
    neu_model.modify_model('parameters', '85', key='syn_decay_numerator')
    neurons = create_neurons(N, neu_model)
    sampled_var = np.clip(rng.normal(106, 10, N), 51, 255)
    neurons.Vm = np.rint(sampled_var)

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

            if nsyn<1:
                pass
            else:
                syn_model = int8CUBA()
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
                                              127)
                        con[-1].weight = np.rint(sampled_var)
                    else:
                        sampled_var = np.clip(rng.normal(w_ex,
                                                         w_ex/10,
                                                         nsyn),
                                              0,
                                              127)
                        con[-1].weight = np.rint(sampled_var)

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
                                          128)
                    con[-1].weight = np.rint(sampled_var)
                    con[-1].namespace['w_factor'] = -1  # -1 in decimal

                    sampled_var = np.clip(rng.normal(delay_in,
                                                     delay_in/2,
                                                     nsyn),
                                          1, np.inf)
                    con[-1].delay = np.rint(sampled_var)*ms

    bg_in  = []
    poisson_pop = []
    syn_model = int8CUBA()
    syn_model.modify_model('parameters', 16, key='weight')
    syn_model.connection['p'] = .03
    for r in range(0, 8):
        poisson_pop.append(PoissonGroup(bg_layer[r], rates=args.bg_freq*Hz))
        bg_in.append(create_synapses(poisson_pop[-1], pop[r], syn_model))

    if args.protocol == 2:
        thal_con = []
        thal_input = []
        # More rate was used (orginal was 120) to elicit comparable activity
        stimulus = TimedArray(np.tile([0 for _ in range(70)]
                                      + [960]
                                      + [0 for _ in range(29)], tsim)*Hz,
                       dt=10.*ms)
        thal_input = PoissonGroup(n_layer[8], rates='stimulus(t)')
        thal_nsyn = []
        for r in range(0, 8):
            sources, targets = generate_connection_indices(thal_input.N,
                                                           pop[r],
                                                           table[r][8])
            if not np.any(sources):
                continue
            thal_nsyn.append(len(sources))
            syn_model.modify_model('connection', sources, key='i')
            syn_model.modify_model('connection', targets, key='j')
            thal_con.append(create_synapses(thal_input, pop[r], syn_model))

    ###########################################################################
    # Creating spike monitors
    ###########################################################################
    smon_net = SpikeMonitor(neurons)


    """ ==================== Running ===================== """
    net = Network(collect())

    net.add(neurons, pop, con, bg_in, poisson_pop)

    if args.protocol == 1:
        net.run(tsim*second, report='stdout')
        device.build(args.code_path)
    elif args.protocol == 2:
        net.add(thal_input, thal_con)

        for i, nsyn in enumerate(thal_nsyn):
            sampled_var = np.clip(rng.normal(w_ex_2, w_ex_2/10, nsyn), 0, 127)
            thal_con[i].weight = np.rint(sampled_var)
        net.run(tsim*second, report='stdout')
        device.build(args.code_path)
    gc.collect()

    """ ==================== Plotting ==================== """
    data = pd.DataFrame({'i': np.array(smon_net.i),
                         't': np.array(smon_net.t/defaultclock.dt)})

    # cortical layer labels: e for excitatory; i for inhibitory
    lname = ['L23e', 'L23i', 'L4e', 'L4i', 'L5e', 'L5i','L6e', 'L6i']

    # number of neurons by layer
    n_layer = [0] + n_layer[:-1]
    l_bins = np.cumsum(n_layer) # cumulative number of neurons by layer

    # grouping spiking times for each neuron
    keys,values = data.sort_values(['i','t']).values.T
    ukeys,index=np.unique(keys,True)
    arrays=np.split(values,index[1:])

    spk_neuron = pd.DataFrame({'i':range(0,N),'t':[[]]*N})
    spk_neuron.iloc[ukeys.astype(int),1] = arrays

    # creating a flag to identify cortical layers
    spk_neuron['layer'] = pd.cut(spk_neuron['i'], l_bins, labels=lname, right=False)
    feather.write_dataframe(spk_neuron, args.save_path + 'spikes.feather')

    Metadata = {'dt': str(defaultclock.dt),
                'duration': str(tsim*second)}
    with open(args.save_path + 'metadata.json', 'w') as f:
        json.dump(Metadata, f)
