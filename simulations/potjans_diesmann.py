""" This code is an adaptation of the ReScience publication made by
    Renan et al. (2017), see original (and complete) implementation in
    https://github.com/shimoura/ReScience-submission/tree/ShimouraR-KamijiNL-PenaRFO-CordeiroVL-CeballosCC-RomaroC-RoqueAC-2017/code/figures
    """

from brian2 import *

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gridspec

import pandas as pd
import numpy as np
import scipy.stats as sc

import sys
import gc

tsim = float(sys.argv[1])   # time of simulation
s = 1000
seed(s)
defaultclock.dt = 0.1*ms

bg = 8.0                    # default value for background rate
g = 4.0                     # default value for inhibitory weight balance
w_ex=87.8

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

# Synapses parameters

d_ex = 1.5*ms       # Excitatory delay
std_d_ex = 0.75*ms  # Std. Excitatory delay
d_in = 0.80*ms      # Inhibitory delay
std_d_in = 0.4*ms   # Std. Inhibitory delay
tau_syn = 0.5*ms    # Post-synaptic current time constant

""" =============== Neuron definitions =============== """
tau_m   = 10.0*ms       # membrane time constant
tau_ref = 2.0*ms        # absolute refractory period
Cm      = 250.0*pF      # membrane capacity
v_r     = -65.0*mV      # reset potential
v_th    = -50.0*mV      # fixed firing threshold

# Leaky integrate-and-fire model equations
# dv/dt: equation 1 from the article
# dI/dt: equation 2 from the article
LIFmodel = '''
    dv/dt = (-v + v_r)/tau_m + (I+Iext)/Cm : volt (unless refractory)
    dI/dt = -I/tau_syn : amp
    Iext : amp
    '''
# Reset condition
resetLIF = '''
    v = v_r
    '''

""" ==================== Networks ==================== """
def PDnet(NeuronGroup, w_ex, g, bg_freq):
    w_ex = w_ex*pA
    std_w_ex = 0.1*w_ex
    bg_layer = array([1600, 1500 ,2100, 1900, 2000, 1900, 2900, 2100])

    pop = [] # Stores NeuronGroups, one for each population
    for r in range(0, 8):
        pop.append(NeuronGroup[nn_cum[r]:nn_cum[r+1]])

    syn_model = '''
                w:amp           # synaptic weight
                '''

    # equations executed only when presynaptic spike occurs:
    # for excitatory connections
    pre_eq = '''
            I_post += w
            '''

    con = [] # Stores connections

    ###########################################################################
    # Connecting neurons
    ###########################################################################
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
                # Excitatory connections
                if (c % 2) == 0:
                    # Synaptic weight from L4e to L2/3e is doubled
                    if c == 2 and r == 0:
                        con.append(Synapses(pop[c], pop[r], model=syn_model, on_pre=pre_eq))
                        con[-1].connect(i = pre_index, j = post_index)
                        con[-1].w = '2.0*clip((w_ex + std_w_ex*randn()),w_ex*0.0, w_ex*inf)'
                    else:
                        con.append(Synapses(pop[c], pop[r], model=syn_model, on_pre=pre_eq))
                        con[-1].connect(i = pre_index, j = post_index)
                        con[-1].w = 'clip((w_ex + std_w_ex*randn()),w_ex*0.0, w_ex*inf)'
                    con[-1].delay = 'clip(d_ex + std_d_ex*randn(), 0.1*ms, d_ex*inf)'

                # Inhibitory connections
                else:
                    con.append(Synapses(pop[c], pop[r], model=syn_model, on_pre=pre_eq))
                    con[-1].connect(i = pre_index, j = post_index)
                    con[-1].w = '-g*clip((w_ex + std_w_ex*randn()),w_ex*0.0, w_ex*inf)'
                    con[-1].delay = 'clip(d_in + std_d_in*randn(), 0.1*ms, d_in*inf)'

    ###########################################################################
    # Creating poissonian background inputs
    ###########################################################################
    bg_in  = []
    for r in range(0, 8):
        bg_in.append(PoissonInput(pop[r], 'I', bg_layer[r], bg_freq*Hz, weight=w_ex))

    ###########################################################################
    # Creating spike monitors
    ###########################################################################
    smon_net = SpikeMonitor(NeuronGroup)

    return pop, con, bg_in, smon_net


""" ==================== Running ===================== """
neurons = NeuronGroup(N, LIFmodel, threshold='v>v_th', reset=resetLIF, \
                        method='linear', refractory=tau_ref)

neurons.v = '-58.0*mV + 10.0*mV*randn()'
neurons.I = 0.0*pA      # initial value for synaptic currents
neurons.Iext = 0.0*pA   # constant external current

pop, con, bg_in, smon_net = PDnet(neurons, w_ex, g, bg)
filename = 'sim_data/PD.dat'

net = Network(collect())

net.add(neurons,pop, con, bg_in)    # Adding objects to the simulation
net.run(tsim*second, report='stdout')

savetxt(filename, c_[smon_net.i,smon_net.t/ms],fmt="%i %.2f")

gc.collect()    #garbage collector to clean memory

""" ==================== Plotting ==================== """
data = pd.read_csv(filename, sep=" ", header=None, names=['i','t'])

# cortical layer labels: e for excitatory; i for inhibitory
lname = ['L23e', 'L23i', 'L4e', 'L4i', 'L5e', 'L5i','L6e', 'L6i']

# number of neurons by layer
n_layer = [0, 20683, 5834, 21915, 5479, 4850, 1065, 14395, 2948];
l_bins = np.cumsum(n_layer) # cumulative number of neurons by layer
N = np.sum(n_layer)         # total number of neurons

# graphs color codes: different colors for different layers
dotsize = 2.5
dotcolor = np.array([[0.0, 0.0, 255.0],
                    [102.0, 178.0, 255.0],
                    [255.0, 128.0, 0.0],
                    [255.0, 178.0, 102.0],
                    [0.0,   128.0, 0.0],
                    [153.0, 255.0, 153.0],
                    [255.0, 0.0,   0.0],
                    [255.0, 153.0, 153.0]])/255.0

# grouping spiking times for each neuron
keys,values = data.sort_values(['i','t']).values.T
ukeys,index=np.unique(keys,True)
arrays=np.split(values,index[1:])

spk_neuron = pd.DataFrame({'i':range(0,N),'t':[[]]*N})
spk_neuron.iloc[ukeys.astype(int),1] = arrays

# creating a flag to identify cortical layers
spk_neuron['layer'] = pd.cut(spk_neuron['i'], l_bins, labels=lname, right=False)
data['layer'] = pd.cut(data['i'], l_bins, labels=lname, right=False)

# sampling data:
psample = 0.025 # percentage of neurons by layer for the raster plot
n_sample = 1000 # number of neurons by layer for sampled measures
spk_neuron = spk_neuron.groupby(['layer'], group_keys=False).apply(lambda x: x.sample(n=n_sample))

# measures DataFrame:
measures_layer = pd.DataFrame(index=lname)

# cleaning variables
del keys, values, ukeys, index, arrays

# figure size for the graphs
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(7.5,10))

###############################################################################
# Raster plot
###############################################################################
plt.subplot2grid((3,2),(0,0), rowspan=3)
plt.gca().set_yticklabels([])
acum_index = 0

for i in range(len(lname)):
    index_start = l_bins[i]
    index_end = l_bins[i]+int(psample*n_layer[i+1])

    x = data.t[data.i.isin(range(index_start,index_end))]
    y = data.i[data.i.isin(range(index_start,index_end))] + acum_index - index_start

    plt.plot(x/1000.0,y,'.',markersize=dotsize,color=dotcolor[i])

    # layers labels
    xpos = tsim-440/1000.0
    ypos = acum_index + (index_end-index_start)/2.0
    plt.text(xpos,ypos,lname[i],horizontalalignment='center', fontweight='bold')

    acum_index = acum_index + (index_end-index_start)

plt.xlim(tsim-400/1000.0,tsim)
plt.ylim(0,acum_index)
plt.xlabel('time [s]')
plt.ylabel(' ')
plt.gca().invert_yaxis()

###############################################################################
# Firing rates
###############################################################################
freq = []
freq = [float(len(spk_neuron.t.iloc[i]))/tsim for i in range(len(spk_neuron))]
spk_neuron['f'] = freq
measures_layer['f'] = spk_neuron.groupby(['layer'])['f'].mean()

# boxplot of firing rates by layer
bplot = spk_neuron.boxplot(column = 'f', by = 'layer', showmeans=True,
                    vert = False, rot = 30, ax = axes[0,1],
                    patch_artist=True, sym='+', return_type='dict', grid=False)

[bplot[0]['boxes'][i].set_facecolor(dotcolor[i]) for i in range(0,len(bplot[0]['boxes']))]
[bplot[0]['means'][i].set_markerfacecolor('white') for i in range(0,len(bplot[0]['boxes']))]
[bplot[0]['means'][i].set_markeredgecolor('k') for i in range(0,len(bplot[0]['boxes']))]

axes[0,1].set_title("")
axes[0,1].set_ylabel("")
axes[0,1].set_xlabel('firing rates[Hz]')
axes[0,1].invert_yaxis()
fig.suptitle("")

###############################################################################
# Interspike intervals + coefficient of variation
###############################################################################
# interspike intervals
isi = []
isi = [np.diff(spk_neuron.t.iloc[i]) for i in range(len(spk_neuron))]

# coefficient of variation
cv = []
cv = [np.std(isi[i])/np.mean(isi[i]) if len(isi[i])>1 else np.nan\
        for i in range(len(spk_neuron))]
spk_neuron['cv'] = cv

measures_layer['cv'] = spk_neuron.groupby(['layer'])['cv'].mean()

# barplot of mean CV
plt.subplot2grid((3,2),(1,1))
measures_layer['cv'].plot.barh(edgecolor='k' ,color=dotcolor, rot=30, width=0.8)
plt.ylabel("")
plt.xlabel('irregularity')
plt.gca().invert_yaxis()

###############################################################################
# Synchrony index
###############################################################################
sync = []
bins = np.arange(0,tsim*1000.0+3.0,3)

for i in range(len(lname)):
    index_sample = spk_neuron[spk_neuron.layer.isin([lname[i]])]
    count, division = np.histogram(data.t[data.i.isin(index_sample)],bins=bins)
    sync.append(np.var(count[166:])/np.mean(count[166:]))

measures_layer['sync'] = sync

# barplot of synchrony index
y_pos = np.arange(len(lname))
plt.subplot2grid((3,2),(2,1))
measures_layer['sync'].plot.barh(edgecolor='k' ,color=dotcolor, rot=30, width=0.8)
plt.ylabel("")
plt.xlabel('synchrony')
plt.gca().invert_yaxis()

plt.tight_layout()
plt.subplots_adjust(left=0.07)
plt.savefig('./sim_data/fig2.pdf', dpi=600)
