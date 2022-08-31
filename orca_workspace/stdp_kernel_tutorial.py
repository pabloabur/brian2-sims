"""
Created on 30.11.2017

@author: Moritz Milde
Email: mmilde@ini.uzh.ch

This script is adapted from https://code.ini.uzh.ch/alpren/gridcells/blob/master/STDP_IE_HaasKernel.py

This script contains a simple event based way to simulate complex STDP kernels
"""

from brian2 import ms, prefs, SpikeMonitor, run
from pyqtgraph.Qt import QtGui
import pyqtgraph as pg
import numpy as np

from teili.core.groups import Neurons, Connections
from teili.models.synapse_models import DPIstdp

from teili.tools.visualizer.DataViewers import PlotSettings
from teili.tools.visualizer.DataModels import StateVariablesModel
from teili.tools.visualizer.DataControllers import Lineplot, Rasterplot

prefs.codegen.target = "numpy"
visualization_backend = 'pyqtgraph'  # Or set it to 'matplotlib' to use matplotlib.pyplot to plot


font = {'family': 'serif',
        'color': 'darkred',
        'weight': 'normal',
        'size': 16,
        }


tmax = 30 * ms
N = 100

# Presynaptic neurons G spike at times from 0 to tmax
# Postsynaptic neurons G spike at times from tmax to 0
# So difference in spike times will vary from -tmax to +tmax
pre_neurons = Neurons(N, model='''tspike:second''', threshold='t>tspike', refractory=100 * ms)

pre_neurons.namespace.update({'tmax': tmax})
post_neurons = Neurons(N, model='''
                Iin0 : amp
                tspike:second''', threshold='t>tspike', refractory=100 * ms)
post_neurons.namespace.update({'tmax': tmax})

pre_neurons.tspike = 'i*tmax/(N-1)'
post_neurons.tspike = '(N-1-i)*tmax/(N-1)'


stdp_synapse = Connections(pre_neurons, post_neurons,
                equation_builder=DPIstdp(), name='stdp_synapse')

stdp_synapse.connect('i==j')

# Setting parameters
stdp_synapse.w_plast = 0.5
stdp_synapse.dApre = 0.01
stdp_synapse.taupre = 10 * ms
stdp_synapse.taupost = 10 * ms


spikemon_pre_neurons = SpikeMonitor(pre_neurons, record=True)
spikemon_post_neurons = SpikeMonitor(post_neurons, record=True)

run(tmax + 1 * ms)


if visualization_backend == 'pyqtgraph':
    app = QtGui.QApplication.instance()
    if app is None:
        app = QtGui.QApplication(sys.argv)
    else:
        print('QApplication instance already exists: %s' % str(app))
else:
    app=None

datamodel = StateVariablesModel(state_variable_names=['w_plast'],
                                state_variables=[stdp_synapse.w_plast],
                                state_variables_times=[np.asarray((post_neurons.tspike - pre_neurons.tspike) / ms)])
Lineplot(DataModel_to_x_and_y_attr=[(datamodel, ('t_w_plast', 'w_plast'))],
        title="Spike-time dependent plasticity",
        xlabel='\u0394 t',  # delta t
        ylabel='w',
        backend=visualization_backend,
        QtApp=app,
        show_immediately=False)

Rasterplot(MyEventsModels=[spikemon_pre_neurons, spikemon_post_neurons],
            MyPlotSettings=PlotSettings(colors=['r']*2),
            title='',
            xlabel='Time (s)',
            ylabel='Neuron ID',
            backend=visualization_backend,
            QtApp=app,
            show_immediately=True)
