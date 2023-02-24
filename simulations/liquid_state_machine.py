from brian2 import SpikeMonitor, StateMonitor, SpikeGeneratorGroup
from brian2 import run, ms, mV, pF
from brian2 import device
from brian2 import TimedArray
from brian2 import defaultclock

from core.utils.misc import decimal2minifloat, minifloat2decimal

from core.equations.neurons.fp8LIF import fp8LIF
from core.equations.synapses.fp8CUBA import fp8CUBA
from core.equations.neurons.LIF import LIF
from core.equations.synapses.hSTDP import hSTDP
from core.equations.neurons.LIFIP import LIFIP
from core.equations.synapses.CUBA import CUBA
from core.equations.synapses.STDP import STDP
from core.equations.synapses.iSTDP import iSTDP
from core.builder.groups_builder import create_synapses, create_neurons
from core.utils.testbench import create_item, create_sequence, create_testbench

import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import json
from random import uniform, sample

import neo
import quantities as q
from elephant import statistics, kernels
from elephant.statistics import isi, cv

from viziphant.statistics import plot_instantaneous_rates_colormesh
from brian2tools import brian_plot, plot_state
import feather

from sklearn import svm, preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, ShuffleSplit


def compute_liquid_states(spikes, sim_times, exp_kernel, dt):
    liquid_states = []
    for spk_trains in spikes:
        conv_spks = np.zeros_like(sim_times)
        if len(spk_trains):
            conv_spks[np.around(spk_trains/dt).astype(int)] = 1
            conv_spks = np.convolve(conv_spks, exp_kernel)
        liquid_states.append(conv_spks[:len(sim_times)])

    return np.array(liquid_states)

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

def train_elm(X_train, y_train, X_test, num_neu):
    X_train = np.matrix(X_train)
    y_train = np.matrix(y_train)
    X_test = np.matrix(X_test)
    n_features = np.shape(X_train)[1]
    weight = np.matrix(np.random.normal(size=(num_neu, n_features)))
    bias = np.matrix(np.random.normal(size=(1, num_neu)))

    # Note broadcast happening here; there is one bias per neuron
    H = sigmoid(X_train*weight.T + bias)
    H_cross = np.linalg.pinv(H)
    beta = H_cross * y_train

    prediction = sigmoid(X_test*weight.T + bias) * beta

    return prediction

def test_elm():
    ''' Function to test performance of ELM used. A simple dataset such as 
    NIST is solved with high accuracy (close to 1).
    '''
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(digits.data,
                                                        digits.target)
    y_test = enc.fit_transform(y_test[:, np.newaxis]).toarray()
    y_train = enc.fit_transform(y_train[:, np.newaxis]).toarray()
    test_size = X_test.shape[0]
    n_hits = [0 for _ in range(1, 11)]
    for i, n_hidden in enumerate(ns_hidden):
        output = train_elm(X_train, y_train, X_test, n_hidden)
        pred = np.array([np.argmax(output[i]) for i in range(test_size)])
        actual = np.array([np.argmax(y_test[i]) for i in range(test_size)])
        n_hits[i] = np.sum(pred==actual)
    plt.plot(ns_hidden, np.array(n_hits)/test_size)
    plt.savefig('./fig_test.png')

def liquid_state_machine(args):
    if args.precision == 'fp8':
        liquid_neu = fp8LIF
        liquid_syn = fp8CUBA
    elif args.precision == 'fp64':
        liquid_neu = LIF
        liquid_syn = CUBA

    defaultclock.dt = args.timestep * ms

    """ =================== Inputs =================== """
    # this for mus silicium
    mus_silic = pd.read_csv(
        'datasets/spikes.csv',
        names=['speaker', 'digit']+[f'ch{i}' for i in range(40)])
    labels = mus_silic.loc[:, 'digit'].values.tolist()
    num_labels = len(np.unique(labels))
    mus_silic = mus_silic.loc[:, ~mus_silic.columns.isin(['speaker', 'digit'])].values.tolist()
    sequences = []
    for spk_t in mus_silic:
        seq_i = [x for x in range(len(spk_t)) if not math.isnan(spk_t[x])]
        seq_t = np.array([x for x in spk_t if str(x) != 'nan']) - np.nanmin(spk_t)
        sequences.append({'times': seq_t, 'indices': seq_i})
    probs = None
    repetitions = len(sequences)

    inter_seq_interval = 200
    input_indices, input_times, events = create_testbench(sequences,
                                                          labels,
                                                          probs,
                                                          inter_seq_interval,
                                                          repetitions)
    input_indices = np.array(input_indices)
    input_times = np.array(input_times) * ms
    num_channels = int(max(input_indices) + 1)
    sim_dur = events[-1][2] + inter_seq_interval*ms
    test_size = 45
    test_t = events[-test_size][2] + inter_seq_interval*ms
    input_spikes = SpikeGeneratorGroup(num_channels,
                                       input_indices,
                                       input_times)

    """ =================== Neurons =================== """
    Nt = args.size
    Ne, Ni = np.rint(Nt*.80).astype(int), np.rint(Nt*.20).astype(int)
    # In case rounding makes a difference
    Nt = Ne + Ni

    e_neu_model = liquid_neu()

    e_neu_model.model += 'x : integer (constant)\ny : integer (constant)\nz : integer (constant)\n'
    if args.precision == 'fp64':
        e_neu_model.modify_model('model',
                               'gtot = gtot0 + gtot1 + gtot2 + gtot3',
                               old_expr='gtot = gtot0')
        e_neu_model.model += 'gtot1 : volt\ngtot2 : volt\ngtot3 : volt\n'
        # changing original parameters to make simulations similar
        e_neu_model.modify_model('namespace', 70*pF, key='Cm')
    if args.precision == 'fp8':
        e_neu_model.modify_model('parameters',
                                 decimal2minifloat(.875),
                                 key='alpha_syn')

    cells = create_neurons(Nt, e_neu_model)

    # Random placement in a grid
    net_grid = np.arange(Nt)
    np.random.shuffle(net_grid)
    # Positions in x distributed in x according to size. y and z are 2x16
    x_dist = int(Nt/32)
    y_dim = 2
    # First operation creates multiples of available grid points
    cells.x = net_grid % x_dist
    # For each previous repetition, same operation to get next points. This is
    # why previous dimension is divided, so it "waits" for x to cycle
    cells.y = (net_grid // x_dist) % y_dim
    # At last, multiple of both previous dimensions is divided
    cells.z = net_grid // (x_dist*y_dim)

    exc_cells = cells[:Ne]
    inh_cells = cells[Ne:]

    """ =================== Connections =================== """
    e_syn_model = liquid_syn()
    e_syn_model.modify_model('connection', .12, key='p')
    if args.precision == 'fp8':
        e_syn_model.modify_model('parameters',
                                 97,
                                 key='weight')
    if args.precision == 'fp64':
        e_syn_model.modify_model('parameters', 23.5*mV, key='weight')
        e_syn_model.modify_model('model', 'gtot1_post', old_expr='gtot0_post')
    thl_conns = create_synapses(input_spikes, cells, e_syn_model)

    e_syn_model = liquid_syn()
    e_syn_model.modify_model(
        'connection',
        '.3 * exp(-((x_pre-x_post)**2 + (y_pre-y_post)**2 + (z_pre-z_post)**2) / 2**2)',
        key='p')
    e_syn_model.modify_model('parameters', '20*rand()*ms', key='delay')
    if args.precision == 'fp64':
        e_syn_model.modify_model('model', 'gtot2_post', old_expr='gtot0_post')
    exc_exc = create_synapses(exc_cells, exc_cells, e_syn_model)

    e_syn_model.modify_model(
        'connection',
        '.2 * exp(-((x_pre-x_post)**2 + (y_pre-y_post)**2 + (z_pre-z_post)**2) / 2**2)',
        key='p')
    exc_inh = create_synapses(exc_cells, inh_cells, e_syn_model)

    i_syn_model = liquid_syn()
    i_syn_model.modify_model(
        'connection',
        '.1 * exp(-((x_pre-x_post)**2 + (y_pre-y_post)**2 + (z_pre-z_post)**2) / 2**2)',
        key='p')
    if args.precision == 'fp8':
        i_syn_model.modify_model('namespace',
                                 decimal2minifloat(-1),
                                 key='w_factor')
    if args.precision == 'fp64':
        i_syn_model.modify_model('namespace', -1, key='w_factor')
        i_syn_model.modify_model('model', 'gtot3_post', old_expr='gtot0_post')
    inh_inh = create_synapses(inh_cells, inh_cells, i_syn_model)

    i_syn_model.modify_model(
        'connection',
        '.4 * exp(-((x_pre-x_post)**2 + (y_pre-y_post)**2 + (z_pre-z_post)**2) / 2**2)',
        key='p')
    inh_exc = create_synapses(inh_cells, exc_cells, i_syn_model)

    """ =================== Results =================== """
    spkmon_e = SpikeMonitor(exc_cells)
    spkmon_i = SpikeMonitor(inh_cells)

    # This just for classify on input directly
    spkmon_inp = SpikeMonitor(input_spikes)

    print('Running simulation')
    run(test_t)

    run(sim_dur-test_t)
    if args.backend == 'cpp_standalone': device.build(args.code_path)

    sim_times = np.arange(0, sim_dur/ms+1, defaultclock.dt/ms)
    exp_kernel=[np.exp(-x/30) for x in range(1000)]
    liquid_states = compute_liquid_states(
        list(spkmon_e.spike_trains().values())
        + list(spkmon_i.spike_trains().values()),
        sim_times,
        exp_kernel,
        defaultclock.dt)

    classifier = svm.SVC(kernel='linear')
    cv = ShuffleSplit(n_splits=5, test_size=test_size)
    end_of_sample_time = [int(lbls_ts[2]/defaultclock.dt) for lbls_ts in events]

    samples = liquid_states[:, end_of_sample_time].T
    labels = [x[0] for x in events]
    score = cross_val_score(classifier, samples, labels, cv=cv).mean()

    kernel = kernels.GaussianKernel(sigma=30*q.ms)
    temp_trains = spkmon_e.spike_trains()
    spk_trains = [neo.SpikeTrain(temp_trains[x]/ms, t_stop=sim_dur/ms, units='ms')
                  for x in temp_trains]
    pop_rates = statistics.instantaneous_rate(spk_trains,
                                              sampling_period=1*q.ms,
                                              kernel=kernel)
    pop_avg_rates = np.mean(pop_rates, axis=1)

    Metadata = {'dt': str(defaultclock.dt),
                'precision': args.precision,
                'size': args.size,
                'trial': args.trial,
                'accuracy': score,
                'duration': str(sim_dur)}
    with open(args.save_path+'/metadata.json', 'w') as f:
        json.dump(Metadata, f)

    input_spikes = pd.DataFrame(
        {'time_ms': input_times/defaultclock.dt,
         'id': input_indices})
    feather.write_dataframe(input_spikes, f'{args.save_path}/input_spikes.feather')

    rec_spikes = pd.DataFrame(
        {'time_ms': np.array(spkmon_e.t/defaultclock.dt),
         'id': np.array(spkmon_e.i)})
    feather.write_dataframe(rec_spikes, f'{args.save_path}/rec_spikes.feather')

    pd_events = np.array([[ev[0], ev[1]/defaultclock.dt, ev[2]/defaultclock.dt] for ev in events])
    pd_events = pd.DataFrame(pd_events, columns=['label', 'tstart_ms', 'tstop_ms'])
    feather.write_dataframe(pd_events, f'{args.save_path}/events_spikes.feather')

    links = pd.DataFrame(
        {'i': np.concatenate((exc_exc.i, exc_inh.i, inh_inh.i+Ne, inh_exc.i+Ne)),
         'j': np.concatenate((exc_exc.j, exc_inh.j+Ne, inh_inh.j+Ne, inh_exc.j))
         })
    feather.write_dataframe(links, f'{args.save_path}/links.feather')
    nodes = pd.DataFrame(
        {'neu_id': [x for x in range(Nt)],
         'type': ['exc' for _ in range(Ne)] + ['inh' for _ in range(Ne, Nt)]})
    feather.write_dataframe(nodes, f'{args.save_path}/nodes.feather')

    liquid_states = compute_liquid_states(
        list(spkmon_inp.spike_trains().values()),
        sim_times,
        exp_kernel,
        defaultclock.dt)
    c_range = np.logspace(-4, 4)
    scores = []
    samples = liquid_states[:, end_of_sample_time].T
    labels = np.reshape([x[0] for x in events], (-1, 1))
    for c in c_range:
        classifier = make_pipeline(svm.SVC(kernel='linear', C=c))
        scores.append(
            cross_val_score(classifier, samples, labels, cv=cv).mean())
    lin_acc = pd.DataFrame({
        'regularization': c_range,
        'score': scores})
    feather.write_dataframe(lin_acc, f'{args.save_path}/linear_acc.feather')

    enc = preprocessing.OneHotEncoder(categories='auto')
    X_train = samples[:-test_size, :]
    y_train = enc.fit_transform(labels[:-test_size, :]).toarray()
    X_test = samples[-test_size:, :]
    y_test = enc.fit_transform(labels[-test_size:, :]).toarray()

    hidden_neu_range = range(1, 41)
    ns_hidden = [i*10 for i in hidden_neu_range]
    n_hits = [0 for _ in hidden_neu_range]
    for i, n_hidden in enumerate(ns_hidden):
        output = train_elm(X_train, y_train, X_test, n_hidden)
        pred = np.array([np.argmax(output[i]) for i in range(test_size)])
        actual = np.array([np.argmax(y_test[i]) for i in range(test_size)])
        n_hits[i] = np.sum(pred==actual)
    elm_acc = pd.DataFrame({
        'size': ns_hidden,
        'score': np.array(n_hits)/test_size})
    feather.write_dataframe(elm_acc, f'{args.save_path}/elm_acc.feather')

    if not args.quiet:
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax2.plot(pop_rates.times, pop_avg_rates, color='red')
        brian_plot(spkmon_e, marker=',', color='black', axes=ax1)
        ax1.set_xlabel(f'time ({pop_rates.times.dimensionality.latex})')
        ax1.set_ylabel('neuron number')
        ax2.set_ylabel(f'rate ({pop_rates.dimensionality})')
        plt.savefig(f'{args.save_path}/fig1.png')

        fig, (ax3, ax4) = plt.subplots(2, 1, sharex=True)
        brian_plot(spkmon_e, axes=ax3)
        ax4.plot(input_times/ms, input_indices, '.')
        plt.savefig(f'{args.save_path}/fig2.png')

        plot_instantaneous_rates_colormesh(pop_rates)
        plt.title('Neuron rates on last trial')
        plt.savefig(f'{args.save_path}/fig3.png')

        plt.figure()
        plt.semilogx(c_range, scores)
        plt.savefig(f'{args.save_path}/fig4.png')

        plt.figure()
        plt.plot(ns_hidden, np.array(n_hits)/test_size)
        plt.savefig(f'{args.save_path}/fig5.png')
