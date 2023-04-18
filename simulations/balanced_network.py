from brian2 import PoissonGroup, SpikeMonitor, StateMonitor
from brian2 import defaultclock, Network, device
from brian2 import Hz, ms, mV

from core.utils.misc import minifloat2decimal, decimal2minifloat
from core.utils.prepare_models import generate_connection_indices
from core.utils.process_responses import neurons_rate

from core.equations.neurons.LIF import LIF
from core.equations.synapses.CUBA import CUBA
from core.equations.neurons.fp8LIF import fp8LIF
from core.equations.synapses.fp8CUBA import fp8CUBA
from core.equations.neurons.int4LIF import int4LIF
from core.equations.synapses.int4CUBA import int4CUBA
from core.equations.neurons.int8LIF import int8LIF
from core.equations.synapses.int8CUBA import int8CUBA
from core.builder.groups_builder import create_synapses, create_neurons

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import feather


# Code using orca column and standard parameters was
# obselete so it was removed.
# It can still be found in predictive learning and
# extrapolation, bbut it should
# be removed as well. This is because I am not creating
# a single motif that is to
# be repeated somewhere else with minor modifications
# anymore. I am rather just
# creating one network for each case. Previous examples
# can be found in previous
# repository from gitlab
def balanced_network(args):
    defaultclock.dt = args.timestep * ms

    """ ==================== Input ==================== """
    poisson_spikes = PoissonGroup(285, rates=6*Hz)

    """ ==================== Models ==================== """
    Ne, Ni = 3471, 613

    # Creating neurons
    neurons, exc_neurons, inh_neurons = [], [], []
    models = zip([LIF, int4LIF, int8LIF, fp8LIF],
                 ['fp64_neu', 'int4_neu', 'int8_neu', 'fp8_neu'])
    for model, name in models:
        aux_model = model()
        if name == 'fp64_neu':
            aux_model.model += 'gtot1 : volt\ngtot2 : volt\n'
            aux_model.modify_model('model', 'gtot = gtot0 + gtot1 + gtot2',
                                   old_expr='gtot = gtot0')
        neurons.append(create_neurons(Ne+Ni,
                                      aux_model,
                                      name=name))
        exc_neurons.append(neurons[-1][:Ne])
        inh_neurons.append(neurons[-1][Ne:])

    # Connecting input
    sources, targets = generate_connection_indices(poisson_spikes.N,
                                                   Ne+Ni,
                                                   0.25)
    thalamus_connections = []
    models = zip([CUBA, int4CUBA, int8CUBA, fp8CUBA],
                 neurons,
                 ['fp64_thal', 'int4_thal', 'int8_thal', 'fp8_thal'])
    for model, neu, name in models:
        aux_model = model()
        aux_model.modify_model('connection', sources, key='i')
        aux_model.modify_model('connection', targets, key='j')
        rng = np.random.default_rng()
        if name == 'fp64_thal':
            aux_model.modify_model('parameters',
                                   rng.normal(25, 2.5, len(sources))*mV,
                                   key='weight')
        elif name == 'int4_thal':
            aux_model.modify_model('parameters',
                                   np.rint(np.clip(rng.normal(1,
                                                              .1,
                                                              len(sources)),
                                                   0, 7)),
                                   key='weight')
        elif name == 'int8_thal':
            aux_model.modify_model('parameters',
                                   np.rint(np.clip(rng.normal(16,
                                                              1.6,
                                                              len(sources)),
                                                   0, 127)),
                                   key='weight')
        elif name == 'fp8_thal':
            aux_model.modify_model('parameters',
                                   decimal2minifloat(rng.normal(36,
                                                                3.6,
                                                                len(sources)),
                                                     raise_warning=False),
                                   key='weight')
        thalamus_connections.append(create_synapses(poisson_spikes,
                                                    neu,
                                                    aux_model,
                                                    name=name))

    # Connecting recurrent synapses
    w_max = {'fp64_syn': 70, 'int4_syn': 8, 'int8_syn': 128, 'fp8_syn': 480}
    inhibitory_weight = {'fp64_syn': None, 'int4_syn': None,
                         'int8_syn': None, 'fp8_syn': None}
    # Excitatory input is the smallest, non-zero positive value of inh weights
    excitatory_weight = {'fp64_syn': 2.1875, 'int4_syn': 1,
                         'int8_syn': 4, 'fp8_syn': 15}
    intra_exc, intra_inh = [], []
    sources_e, targets_e = generate_connection_indices(Ne, neu.N, .1)
    sources_i, targets_i = generate_connection_indices(Ni, neu.N, .1)
    models = zip([CUBA, int4CUBA, int8CUBA, fp8CUBA],
                 exc_neurons,
                 inh_neurons,
                 neurons,
                 ['fp64_syn', 'int4_syn', 'int8_syn', 'fp8_syn'])
    for model, e_neu, i_neu, neu, name in models:
        # Set excitatory weights
        aux_model = model()
        aux_model.modify_model('connection', sources_e, key='i')
        aux_model.modify_model('connection', targets_e, key='j')
        if name == 'fp64_syn':
            aux_model.modify_model('model',
                                   'gtot1_post',
                                   old_expr='gtot0_post')
            aux_model.modify_model('parameters',
                                   rng.normal(excitatory_weight[name],
                                              excitatory_weight[name]/10,
                                              len(sources_e))*mV,
                                   key='weight')
        elif name == 'fp8_syn':
            aux_model.modify_model('parameters',
                                   decimal2minifloat(
                                       rng.normal(excitatory_weight[name],
                                                  excitatory_weight[name]/10,
                                                  len(sources_e)),
                                       raise_warning=False),
                                   key='weight')
        elif name == 'int4_syn':
            aux_model.modify_model('parameters',
                                   np.rint(np.clip(
                                       rng.normal(excitatory_weight[name],
                                                  excitatory_weight[name]/10,
                                                  len(sources_e)),
                                           0, 7)),
                                   key='weight')
        elif name == 'int8_syn':
            aux_model.modify_model('parameters',
                                   np.rint(np.clip(
                                       rng.normal(excitatory_weight[name],
                                                  excitatory_weight[name]/10,
                                                  len(sources_e)),
                                           0, 127)),
                                   key='weight')
        intra_exc.append(create_synapses(e_neu,
                                         neu,
                                         aux_model,
                                         name=name+'_e'))

        # Set negative factor of inhibitory weights
        aux_model = model()
        aux_model.modify_model('connection', sources_i, key='i')
        aux_model.modify_model('connection', targets_i, key='j')
        if name == 'fp64_syn':
            aux_model.modify_model('model',
                                   'gtot2_post',
                                   old_expr='gtot0_post')

        if name == 'fp8_syn':
            aux_model.modify_model('namespace',
                                   decimal2minifloat(-1),
                                   key='w_factor')
        else:
            aux_model.modify_model('namespace', -1, key='w_factor')

        # Calculates percentage of maximum weight for each case
        inhibitory_weight[name] = args.w_perc*w_max[name]
        if name == 'fp64_syn':
            aux_model.modify_model('parameters',
                                   rng.normal(inhibitory_weight[name],
                                              inhibitory_weight[name]/10,
                                              len(sources_i))*mV,
                                   key='weight')
        elif name == 'fp8_syn':
            aux_model.modify_model('parameters',
                                   decimal2minifloat(
                                       rng.normal(inhibitory_weight[name],
                                                  inhibitory_weight[name]/10,
                                                  len(sources_i)),
                                       raise_warning=False),
                                   key='weight')
        elif name == 'int4_syn':
            aux_model.modify_model(
                'parameters',
                np.rint(
                    np.clip(rng.normal(inhibitory_weight[name],
                                       inhibitory_weight[name]/10,
                                       len(sources_i)),
                            0, 7)),
                key='weight')
        elif name == 'int8_syn':
            aux_model.modify_model(
                'parameters',
                np.rint(
                    np.clip(rng.normal(inhibitory_weight[name],
                                       inhibitory_weight[name]/10,
                                       len(sources_i)),
                            0, 127)),
                key='weight')

        intra_inh.append(create_synapses(i_neu,
                                         neu,
                                         aux_model,
                                         name=name+'_i'))

    """ ==================== Monitors ==================== """
    selected_exc_cells = rng.choice(Ne, 4, replace=False)
    selected_inh_cells = rng.choice(Ni, 4, replace=False)

    spkmon_e = [SpikeMonitor(x, name=x.name+'_e_spkmon') for x in exc_neurons]
    spkmon_i = [SpikeMonitor(x, name=x.name+'_i_spkmon') for x in inh_neurons]
    sttmon_e = [StateMonitor(x, variables='Vm',
                             record=selected_exc_cells,
                             name=x.name+'_e_sttmon')
                for x in exc_neurons]
    sttmon_i = [StateMonitor(x, variables='Vm',
                             record=selected_inh_cells,
                             name=x.name+'_i_sttmon')
                for x in inh_neurons]

    """ ==================== running/processing ==================== """
    duration = 1000
    net = Network()
    net.add(neurons, exc_neurons, inh_neurons, thalamus_connections, intra_exc,
            intra_inh, poisson_spikes, spkmon_e, spkmon_i, sttmon_e, sttmon_i)
    net.run(duration*ms, namespace={}, report='stdout', report_period=100*ms)
    if args.backend == 'cpp_standalone':
        device.build(args.code_path)

    population_rates = [neurons_rate(x, duration, sigma=50) for x in spkmon_e]
    pop_avg_rates = [np.mean(x, axis=1) for x in population_rates]

    """ ==================== saving results ==================== """
    Metadata = {'selected_exc_cells': selected_exc_cells.tolist(),
                'selected_inh_cells': selected_inh_cells.tolist(),
                'dt': str(defaultclock.dt),
                'trial': args.trial,
                'duration': str(duration*ms),
                'inh_perc': args.w_perc}
    with open(args.save_path + '/metadata.json', 'w') as f:
        json.dump(Metadata, f)

    # Prepares to save final mean inhibitory weight for each case
    resolution = ['fp64', 'int4', 'int8', 'fp8']
    for res in resolution:
        if res == 'fp8':
            inhibitory_weight[res+'_syn'] = minifloat2decimal(
                    decimal2minifloat(inhibitory_weight[res+'_syn']))[0]
        elif res == 'int8':
            inhibitory_weight[res+'_syn'] = np.rint(
                inhibitory_weight[res+'_syn'])
        elif res == 'int4':
            inhibitory_weight[res+'_syn'] = np.rint(
                inhibitory_weight[res+'_syn'])

    avg_rates = pd.DataFrame({
        'inhibitory_weight': list(inhibitory_weight.values()),
        'resolution': resolution,
        'frequency_Hz': [np.mean(x.magnitude) for x in pop_avg_rates]})
    feather.write_dataframe(avg_rates, args.save_path + '/avg_rates.feather')

    voltages = pd.DataFrame({
        'values': np.hstack((sttmon_e[0].Vm[2]/(20*mV),
                             minifloat2decimal(sttmon_e[3].Vm[2])/480)),
        'time_ms': np.hstack((sttmon_e[0].t, sttmon_e[3].t)),
        'resolution': (['fp64' for _ in range(len(sttmon_e[0].t))]
                       + ['fp8' for _ in range(len(sttmon_e[3].t))])})
    feather.write_dataframe(voltages, args.save_path + '/voltages.feather')

    weights = pd.DataFrame({
        'values': np.hstack((intra_inh[0].weight/mV,
                             np.array(intra_inh[1].weight),
                             np.array(intra_inh[2].weight),
                             minifloat2decimal(intra_inh[3].weight))),
        'resolution': (['fp64' for _ in range(len(intra_inh[0].weight))]
                       + ['int4' for _ in range(len(intra_inh[1].weight))]
                       + ['int8' for _ in range(len(intra_inh[2].weight))]
                       + ['fp8' for _ in range(len(intra_inh[3].weight))])})
    feather.write_dataframe(weights, args.save_path + '/weights.feather')

    if not args.quiet:
        plt.plot(pop_avg_rates[0], label='fp64')
        plt.plot(pop_avg_rates[1], label='int4')
        plt.plot(pop_avg_rates[2], label='int8')
        plt.plot(pop_avg_rates[3], label='fp8')
        plt.legend()
        plt.savefig(f'{args.save_path}/fig1')

        plt.figure()
        plt.plot(sttmon_e[0].Vm[2]/(20*mV))
        plt.plot(minifloat2decimal(sttmon_e[3].Vm[2])/480)
        plt.savefig(f'{args.save_path}/fig2')
