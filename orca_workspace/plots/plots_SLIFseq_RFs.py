from bokeh.io import curdoc
from bokeh.models import CustomJS, Select, Slider, ColumnDataSource, BoxAnnotation
from bokeh.layouts import grid
from bokeh.plotting import figure

import numpy as np
import pickle

import sys

data_folder = sys.argv[1]
sort_type = sys.argv[2]

# Load metadata of given simulation
with open(f'{data_folder}general.data', 'rb') as f:
    metadata = pickle.load(f)
num_inh = metadata['num_inh']
input_rate = metadata['input_rate']
ei_conn = metadata['e->i p']
#ffi_weight = metadata['']
num_exc = metadata['num_exc']
num_channels = metadata['num_channels']
rasters = np.load(f'{data_folder}rasters.npz')
traces = np.load(f'{data_folder}traces.npz')
matrices = np.load(f'{data_folder}matrices.npz', allow_pickle=True)
rf = matrices['rf']
rf = np.reshape(rf, (num_channels, num_exc, -1))
source = ColumnDataSource(data=dict(weights=rf[:, 0, 0],
                                    channels=range(np.shape(rf)[0]),
                                    colors=48*['red']+48*['green']+48*['blue']
                                    ))

# Plots
p = figure(plot_width=900, plot_height=400, x_range=(0, 143), y_range=(0, 16))
p.vbar(x='channels', top='weights', width=0.8, color='colors', source=source)

# Create widgets
neuron_select = Select(title="Neuron:", value="0", options=[str(x) for x in range(num_exc)])
time_slider = Slider(start=0, end=np.shape(rf)[2]-1, value=0, step=1, title="Time (ms)")

def update_plot(attr, old, new):
    source.data = dict(weights=rf[:, int(neuron_select.value), time_slider.value],
                       channels=range(np.shape(rf)[0]),
                       colors=48*['red']+48*['green']+48*['blue']
                       )

time_slider.on_change('value', update_plot)
neuron_select.on_change('value', update_plot)

layout = grid([
        [neuron_select, p],
        [time_slider]
    ])

curdoc().add_root(layout)
