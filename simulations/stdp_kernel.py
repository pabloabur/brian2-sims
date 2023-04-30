"""
This script contains a simple event based way to simulate a stochastic
STDP kernel.
"""

from brian2 import ms, prefs, StateMonitor, SpikeMonitor, run, defaultclock,\
        ExplicitStateUpdater, TimedArray
from pyqtgraph.Qt import QtGui
import pyqtgraph as pg
import numpy as np
import os

from teili.core.groups import Neurons, Connections
from teili.models.synapse_models import QuantStochSynStdp as plastic_synapse_model
from teili.tools.add_run_reg import add_lfsr
from teili.models.builder.synapse_equation_builder import SynapseEquationBuilder

from teili.tools.visualizer.DataViewers import PlotSettings
from teili.tools.visualizer.DataModels import StateVariablesModel
from teili.tools.visualizer.DataControllers import Lineplot, Rasterplot
from teili.tools.lfsr import create_lfsr

prefs.codegen.target = "numpy"
defaultclock.dt = 1 * ms
visualization_backend = 'pyqtgraph'

font = {'family': 'serif',
        'color': 'darkred',
        'weight': 'normal',
        'size': 16,
        }

plastic_synapse_model=SynapseEquationBuilder(base_unit='quantized',
                                 plasticity='quantized_stochastic_stdp',
                                 #random_generator='lfsr_syn',
                                 structural_plasticity='stochastic_counter')

trials = 105
trial_duration = 60
N = trial_duration
wait_time = 2*trial_duration  # set delay between trials to avoid interferences
tmax = trial_duration*trials + wait_time*trials

# Define matched spike times between pre and post neurons
post_tspikes = np.arange(1, N*trials + 1).reshape((trials, N))
pre_tspikes = post_tspikes[:, np.array(range(N-1, -1, -1))]
# Use the ones below to test simultaneous samples from random function
#post_tspikes = np.arange(0, trials, 2).reshape(-1, 1) + np.ones(N)
#pre_tspikes = np.arange(1, trials, 2).reshape(-1, 1) + np.ones(N)

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
    pre_neurons = Neurons(N, model='v = tapre(t, i) : 1',
                          threshold='v == 1', refractory='1*ms')
    pre_neurons.namespace.update({'tmax': tmax})
    pre_neurons.namespace.update({'tapre': tapre})

    post_neurons = Neurons(N, model='''v = tapost(t, i) : 1
                                       Iin0 : amp
                                       I_syn : amp''',
                           threshold='v == 1', refractory='1*ms')
    post_neurons.namespace.update({'tmax': tmax})
    post_neurons.namespace.update({'tapost': tapost})

    stochastic_decay = ExplicitStateUpdater('''x_new = f(x,t)''')
    stdp_synapse = Connections(pre_neurons, post_neurons,
                               method=stochastic_decay,
                               equation_builder=plastic_synapse_model(),
                               name='stdp_synapse')
    stdp_synapse.connect('i==j')

    # Setting parameters
    init_wplast = 7
    stdp_synapse.w_plast = init_wplast
    stdp_synapse.taupre = 20*ms
    stdp_synapse.taupost = 20*ms
    stdp_synapse.stdp_thres = 1
    stdp_synapse.rand_num_bits_Apre = 5
    stdp_synapse.rand_num_bits_Apost = 5

    # Parameters for LFSR
    if 'lfsr' in plastic_synapse_model().keywords['model']:
        num_bits = 5
        stdp_synapse.lfsr_num_bits_syn = num_bits
        stdp_synapse.lfsr_num_bits_Apre = 5
        stdp_synapse.lfsr_num_bits_Apost = 5
        ta = create_lfsr([], [stdp_synapse], defaultclock.dt)

    spikemon_pre_neurons = SpikeMonitor(pre_neurons, record=True)
    spikemon_post_neurons = SpikeMonitor(post_neurons, record=True)
    statemon_synapse = StateMonitor(stdp_synapse,
                                    variables=['Apre', 'Apost', 'w_plast',
                                               'decay_probability_Apre',
                                               'decay_probability_Apost',
                                               're_init_counter'],
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
