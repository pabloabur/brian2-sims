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
import feather
import json

import gc
from datetime import datetime

from core.equations.neurons.fp8LIF import fp8LIF
from core.equations.synapses.fp8CUBA import fp8CUBA
from core.builder.groups_builder import create_synapses, create_neurons
from core.utils.misc import minifloat2decimal, decimal2minifloat


def fp8_potjans_diesmann(protocol, bg_freq, w_in, defaultclock, save_path, code_path):
    if protocol == 1:
        tsim = 60
    elif protocol == 2:
        tsim = 100

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

    # Original excitatory weights were 87.8pA with std 8.78pA. Values below were
    # chosen so as to ensure good activity under input adopted.
    w_ex = 58
    w_ex_2 = 66

    """ =============== Neuron definitions =============== """
    fp8_values = range(128)
    neu_model = fp8LIF()
    # tau_syn=0.5ms, resulting in decimal alpha_syn of 0.333. Approximated to 0.34375
    # Original tau_m=10ms approximated to 0.9375, i.e. tau=15ms
    neu_model.modify_model('parameters', '43', key='alpha_syn')
    neu_model.modify_model('parameters', '55', key='alpha')
    # Set Vreset so as refractory period is 2ms
    neu_model.modify_model('parameters', '178', key='Vreset')
    neurons = create_neurons(N, neu_model)
    sampled_var = np.rint(np.clip(96 + 30*np.random.randn(N),
                                  min(fp8_values),
                                  max(fp8_values)))
    neurons.Vm = sampled_var

    """ ==================== Networks ==================== """
    bg_layer = array([1600, 1500 ,2100, 1900, 2000, 1900, 2900, 2100])

    pop = [] # Stores NeuronGroups, one for each population
    for r in range(0, 8):
        pop.append(neurons[nn_cum[r]:nn_cum[r+1]])

    con = [] # Stores connections
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
                syn_model = fp8CUBA()
                syn_model.modify_model('connection', pre_index, key='i')
                syn_model.modify_model('connection', post_index, key='j')
                con.append(create_synapses(pop[c], pop[r], syn_model))
                # Excitatory connections
                if (c % 2) == 0:
                    # Synaptic weight from L4e to L2/3e is doubled
                    if c == 2 and r == 0:
                        # ranging between 16 and 24 in decimal
                        sampled_var = np.rint(np.clip(w_ex_2 + 10*np.random.randn(nsyn),
                                                      min(fp8_values),
                                                      max(fp8_values)))
                        con[-1].weight = sampled_var
                    else:
                        # ranging between 8 and 12 in decimal
                        sampled_var = np.rint(np.clip(w_ex + 10*np.random.randn(nsyn),
                                                      min(fp8_values),
                                                      max(fp8_values)))
                        con[-1].weight = sampled_var
                    sampled_var = np.rint(np.clip(1 + 2*randn(nsyn), 1, np.inf))
                    con[-1].delay = sampled_var*ms

                # Inhibitory connections
                else:
                    # ranging between 32 and 52 in decimal (protocol 2)
                    sampled_var = np.rint(np.clip(w_in + 10*np.random.randn(nsyn),
                                                  min(fp8_values),
                                                  max(fp8_values)))
                    con[-1].weight = sampled_var
                    con[-1].namespace['w_factor'] = 184  # -1 in decimal
                    sampled_var = np.rint(np.clip(1 + 1*randn(nsyn), 1, np.inf))
                    con[-1].delay = sampled_var*ms

    bg_in  = []
    poisson_pop = []
    syn_model = fp8CUBA()
    syn_model.connection['p'] = .03
    for r in range(0, 8):
        poisson_pop.append(PoissonGroup(bg_layer[r], rates=bg_freq*Hz))
        bg_in.append(create_synapses(poisson_pop[-1], pop[r], syn_model))

    if protocol == 2:
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
            conn_mat = np.random.choice([0, 1],
                                        size=(thal_input.N, pop[r].N),
                                        p=[1-table[r][8], table[r][8]])
            sources, targets = conn_mat.nonzero()
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

    if protocol == 1:
        net.run(tsim*second, report='stdout')
        device.build(code_path)
    elif protocol == 2:
        net.add(thal_input, thal_con)

        for i, nsyn in enumerate(thal_nsyn):
            sampled_var = np.rint(np.clip(w_ex_2 + 10*np.random.randn(nsyn),
                                          min(fp8_values),
                                          max(fp8_values)))
            thal_con[i].weight = sampled_var
        net.run(tsim*second, report='stdout')
        device.build(code_path)
    gc.collect()

    """ ==================== Plotting ==================== """
    data = pd.DataFrame({'i': np.array(smon_net.i),
                         't': np.array(smon_net.t/defaultclock.dt)})

    # cortical layer labels: e for excitatory; i for inhibitory
    lname = ['L23e', 'L23i', 'L4e', 'L4i', 'L5e', 'L5i','L6e', 'L6i']

    # number of neurons by layer
    n_layer = [0] + n_layer[:-1]
    l_bins = np.cumsum(n_layer) # cumulative number of neurons by layer

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
    feather.write_dataframe(spk_neuron, save_path + 'spikes.feather')

    Metadata = {'dt': str(defaultclock.dt),
                'duration': str(tsim*second)}
    with open(save_path + 'metadata.json', 'w') as f:
        json.dump(Metadata, f)

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

    if protocol == 1:
        plt.xlim(tsim-400/1000.0, tsim)
    elif protocol == 2:
        plt.xlim(600, 800)
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
        index_sample = spk_neuron.i[spk_neuron.layer.isin([lname[i]])]
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

    ########################################################################
    # AIness measure: f<30Hz & 0.7<=cv<1.2 & sync_index <= 8
    ########################################################################
    measures_layer['AI'] = (measures_layer.f<30)&(measures_layer.sync<=8)&\
                            (measures_layer.cv>=0.7)&(measures_layer.cv<1.2)

    # % of layers in the AIness range
    ainess = 100*sum(measures_layer.AI)/8.0

    measures_layer.to_csv(save_path + f'measures_prot{protocol}_bg{bg_freq}_win{w_in}.csv')

    plt.tight_layout()
    plt.subplots_adjust(left=0.07)
    plt.savefig(save_path + f'fig_plots_prot{protocol}_bg{bg_freq}_win{w_in}.pdf', dpi=600)

    ###############################################################################
    # Population spike counts
    ###############################################################################
    if protocol == 2:
        spk_count = []
        bins = np.arange(0,tsim*1000.0+.5,.5)

        for i in range(len(lname)):
            index_start = l_bins[i]
            index_end = l_bins[i]+int(psample*n_layer[i+1])
            count, division = np.histogram(data.t[data.i.isin(np.arange(index_start,index_end))],bins=bins)
            count = np.sum(np.split(count, tsim), axis=0) / float(tsim)
            spk_count.append([count])

        measures_layer['spk_count'] = spk_count
        date_time = datetime.now()
        measures_layer.to_csv(
            save_path
            + f"""hist_{date_time.strftime('%Y.%m.%d')}_{date_time.hour}.{date_time.minute}.csv""")

        for i in range(0,len(lname),2):
            plt.subplot2grid((4, 3), (int(i/2), 1), colspan=2)
            plt.step(np.arange(-10,30,0.5), measures_layer.spk_count[i][0][1380:1460], label=lname[i])
            plt.step(np.arange(-10,30,0.5), measures_layer.spk_count[i+1][0][1380:1460], label=lname[i+1])
            plt.legend()
            if i != 6:
                ax = plt.gca()
                ax.xaxis.set_visible(False)
            else:
                ax = plt.gca()
                ax.yaxis.set_visible(True)
            plt.ylim(0, 15)

        plt.tight_layout()
        plt.subplots_adjust(left=0.07)
        plt.savefig(save_path + f'fig_hist.pdf', dpi=600)
