""" This file has modified versions of some code from Teili. See
M. Milde, A. Renner, R. Krause, A. M. Whatley, S. Solinas, D. Zendrikov,
N. Risi, M. Rasetto, K. Burelo, V. R. C. Leite. teili: A toolbox for building
and testing neural algorithms and computational primitives using spiking
neurons. Unreleased software, Institute of Neuroinformatics, University of
Zurich and ETH Zurich, 2018.
"""

from brian2 import ms, prefs, StateMonitor, SpikeMonitor, run, defaultclock,\
        ExplicitStateUpdater, TimedArray
import numpy as np

import os, sys
sys.path.append(os.getcwd())

from core.utils.misc import minifloat2decimal, decimal2minifloat

from core.equations.neurons.fp8LIF import fp8LIF
from core.equations.synapses.fp8CUBA import fp8CUBA
from core.equations.synapses.fp8STDP import fp8STDP
from core.builder.groups_builder import create_synapses, create_neurons


def stdp_kernel(args):
    trials = 105
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

    average_trials = 100
    average_wplast = np.zeros((average_trials, trial_duration))
    average_counter = np.zeros((average_trials, trial_duration))
    for avg_trial in range(average_trials):
        neuron_model = fp8LIF()
        neuron_model.modify_model(
            'model',
            'decay_term = tapre(t, i)',
            old_expr='decay_term = fp8_multiply(Vm, alpha)')
        neuron_model.modify_model('threshold', '1', old_expr='Vthr')
        neuron_model.namespace = {**neuron_model.namespace,
                                  'tmax': tmax,
                                  'tapre': tapre}
        pre_neurons = create_neurons(N, neuron_model)

        neuron_model = fp8LIF()
        neuron_model.modify_model(
            'model',
            'decay_term = tapost(t, i)',
            old_expr='decay_term = fp8_multiply(Vm, alpha)')
        neuron_model.modify_model('threshold', '1', old_expr='Vthr')
        neuron_model.namespace = {**neuron_model.namespace,
                                  'tmax': tmax,
                                  'tapost': tapost}
        post_neurons = create_neurons(N, neuron_model)

        stdp_model = fp8STDP()
        stdp_model.modify_model('connection', "i==j", key='condition')
        stdp_model.modify_model('parameters',
                                decimal2minifloat(0.03125),
                                key='w_plast')
        stdp_synapse = create_synapses(pre_neurons,
                                       post_neurons,
                                       stdp_model)

        spikemon_pre_neurons = SpikeMonitor(pre_neurons, record=True)
        spikemon_post_neurons = SpikeMonitor(post_neurons, record=True)
        statemon_pre_neurons = StateMonitor(pre_neurons,
                                            variables=['Ca'],
                                            record=True)
        statemon_post_neurons = StateMonitor(post_neurons,
                                             variables=['Ca'],
                                             record=True)
        statemon_synapse = StateMonitor(stdp_synapse,
                                        variables=['w_plast'],
                                        record=True,
                                        name='statemon_synapse')

        run(tmax*ms)
        average_wplast[avg_trial, :] = np.array(stdp_synapse.w_plast)
        average_counter[avg_trial, :] = np.array(stdp_synapse.re_init_counter)

        if visualization_backend == 'pyqtgraph':
            app = QtGui.QApplication.instance()
            if app is None:
                app = QtGui.QApplication(sys.argv)
            else:
                print('QApplication instance already exists: %s' % str(app))
        else:
            app = None

    pairs_timing = (spikemon_post_neurons.t[:trial_duration]
                    - spikemon_post_neurons.t[:trial_duration][::-1])/ms
    win_2 = pg.GraphicsWindow(title="trials")
    datamodel = StateVariablesModel(state_variable_names=['w_plast'],
                                    state_variables=[average_wplast[avg_trial-1, :]-init_wplast],
                                    state_variables_times=[pairs_timing])
    Lineplot(DataModel_to_x_and_y_attr=[(datamodel, ('t_w_plast', 'w_plast'))],
            title="Spike-time dependent plasticity (trial)",
            xlabel='\u0394 t (ms)',  # delta t
            ylabel='\u0394 w',
            backend=visualization_backend,
            QtApp=app,
            mainfig=win_2,
            show_immediately=False)

    datamodel = StateVariablesModel(state_variable_names=['re_init_counter'],
                                    state_variables=[average_counter[avg_trial-1, :]],
                                    state_variables_times=[pairs_timing])
    Lineplot(DataModel_to_x_and_y_attr=[(datamodel, ('t_re_init_counter', 're_init_counter'))],
            title="Homeostatic counter (trial)",
            xlabel='\u0394 t (ms)',  # delta t
            ylabel='counter value',
            backend=visualization_backend,
            QtApp=app,
            mainfig=win_2,
            show_immediately=False)

    win_5 = pg.GraphicsWindow(title="averages")
    datamodel = StateVariablesModel(state_variable_names=['w_plast'],
                                    state_variables=[np.mean(average_wplast, axis=0)-init_wplast],
                                    state_variables_times=[pairs_timing])
    Lineplot(DataModel_to_x_and_y_attr=[(datamodel, ('t_w_plast', 'w_plast'))],
            title="Spike-time dependent plasticity (average)",
            xlabel='\u0394 t (ms)',  # delta t
            ylabel='\u0394 w',
            backend=visualization_backend,
            QtApp=app,
            mainfig=win_5,
            show_immediately=False)

    datamodel = StateVariablesModel(state_variable_names=['re_init_counter'],
                                    state_variables=[np.mean(average_counter, axis=0)],
                                    state_variables_times=[pairs_timing])
    Lineplot(DataModel_to_x_and_y_attr=[(datamodel, ('t_re_init_counter', 're_init_counter'))],
            title="Homeostatic counter (average)",
            xlabel='\u0394 t (ms)',  # delta t
            ylabel='\u0394 w',
            backend=visualization_backend,
            QtApp=app,
            mainfig=win_5,
            show_immediately=False)

    win_6 = pg.GraphicsWindow(title="Spikes")
    Rasterplot(MyEventsModels=[spikemon_pre_neurons, spikemon_post_neurons],
                MyPlotSettings=PlotSettings(colors=['w', 'r']),
                title='',
                xlabel='Time (s)',
                ylabel='Neuron ID',
                backend=visualization_backend,
                QtApp=app,
                mainfig=win_6,
                show_immediately=False)

    win_7 = pg.GraphicsWindow(title='As weak depot.')
    Lineplot(DataModel_to_x_and_y_attr=[(statemon_synapse[8], ('t', 'Apre')), (statemon_synapse[8], ('t', 'Apost'))],
            title="Apre",
            xlabel='time',  # delta t
            ylabel='Apre',
            backend=visualization_backend,
            QtApp=app,
            mainfig=win_7,
            show_immediately=False)

    win_8 = pg.GraphicsWindow(title='As strong depot.')
    Lineplot(DataModel_to_x_and_y_attr=[(statemon_synapse[25], ('t', 'Apre')), (statemon_synapse[25], ('t', 'Apost'))],
            title="Apre",
            xlabel='time',  # delta t
            ylabel='Apre',
            backend=visualization_backend,
            QtApp=app,
            mainfig=win_8,
            show_immediately=False)

    win_9 = pg.GraphicsWindow(title='As weak pot.')
    Lineplot(DataModel_to_x_and_y_attr=[(statemon_synapse[49], ('t', 'Apre')), (statemon_synapse[49], ('t', 'Apost'))],
            title="Apre",
            xlabel='time',  # delta t
            ylabel='Apre',
            backend=visualization_backend,
            QtApp=app,
            mainfig=win_9,
            show_immediately=True)

stdp_kernel(0)
