from brian2 import SpikeMonitor, StateMonitor, SpikeGeneratorGroup
from brian2 import run, ms, mV, pF
from brian2 import device
from brian2 import TimedArray

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
import git
import sys
sys.path.extend([git.Repo('.').git.rev_parse('--show-toplevel')])

import neo
import quantities as q
from elephant import statistics, kernels
from elephant.statistics import isi, cv
from elephant.conversion import BinnedSpikeTrain

from viziphant.statistics import plot_instantaneous_rates_colormesh
from brian2tools import brian_plot, plot_state
import feather

from sklearn.svm import LinearSVC


def liquid_state_machine(size, precision, defaultclock, trial_no, path, quiet):
    # freezing noise
    #import random
    #random.seed(25)
    #from brian2 import seed
    #np.random.seed(25)
    #seed(25)

    if precision == 'fp8':
        liquid_neu = fp8LIF
        liquid_syn = fp8CUBA
    elif precision == 'fp64':
        liquid_neu = LIF
        liquid_syn = CUBA

    """ =================== Inputs =================== """
    item_rate = 128
    repetitions = 140
    inter_spk_interval = np.ceil(1/item_rate*1000).astype(int)
    inter_seq_interval = 200
    item_spikes = 1
    A = create_item([0], inter_spk_interval, item_spikes)
    B = create_item([1], inter_spk_interval, item_spikes)
    C = create_item([2], inter_spk_interval, item_spikes)
    D = create_item([3], inter_spk_interval, item_spikes)
    E = create_item([4], inter_spk_interval, item_spikes)
    F = create_item([5], inter_spk_interval, item_spikes)
    G = create_item([6], inter_spk_interval, item_spikes)
    H = create_item([7], inter_spk_interval, item_spikes)

    seq1 = [A, B, C, D, E, F, G, H]
    seq2 = [H, G, F, E, D, C, B, A]
    seq1 = create_sequence(seq1, inter_spk_interval)
    seq2 = create_sequence(seq2, inter_spk_interval)
    sequences = [seq1, seq2]

    num_labels = 2
    labels = [x for x in range(num_labels)]
    probs = [.5, .5]

    # this for mus silicium
    mus_silic = pd.read_csv(
        'spikes.csv', names=['speaker', 'digit']+[f'ch{i}' for i in range(40)])
    labels = mus_silic.loc[:, 'digit'].values.tolist()
    num_labels = len(np.unique(labels))
    mus_silic = mus_silic.loc[:, ~mus_silic.columns.isin(['speaker', 'digit'])].values.tolist()
    #mus_silic = [mus_silic[i] for i, x in enumerate(labels) if x==0 or x==1]
    #labels = [x for i, x in enumerate(labels) if x==0 or x==1]
    sequences = []
    for spk_t in mus_silic:
        seq_i = [x for x in range(len(spk_t)) if not math.isnan(spk_t[x])]
        seq_t = np.array([x for x in spk_t if str(x) != 'nan']) - np.nanmin(spk_t)
        sequences.append({'times': seq_t, 'indices': seq_i})
    probs = None
    repetitions = len(sequences)

    silence = None

    # for emulating sleep
    #sleep_iter = 50
    #silence = {'iteration': [sleep_iter], 'duration': [10000]}

    input_indices, input_times, events = create_testbench(sequences,
                                                          labels,
                                                          probs,
                                                          inter_seq_interval,
                                                          repetitions,
                                                          silence)
    input_indices = np.array(input_indices)
    input_times = np.array(input_times) * ms
    num_channels = int(max(input_indices) + 1)
    sim_dur = events[-1][2] + inter_seq_interval*ms
    test_size = 20
    test_t = events[-test_size][2] + inter_seq_interval*ms
    input_spikes = SpikeGeneratorGroup(num_channels,
                                       input_indices,
                                       input_times)

    """ =================== Neurons =================== """
    Nt = size
    Ne, Ni = np.rint(Nt*.85).astype(int), np.rint(Nt*.15).astype(int)
    # In case rounding makes a difference
    Nt = Ne + Ni

    e_neu_model = liquid_neu()

    # noise
    #rand_samples = [uniform(0, 1)
    #                for _ in range(int((sim_dur/defaultclock.dt)*Nt))]
    #rand_samples = np.reshape(rand_samples, (int(sim_dur/defaultclock.dt), Nt))
    #noise = TimedArray(rand_samples*mV, dt=defaultclock.dt)
    #e_neu_model.modify_model('model', 'alpha*Vm + noise(t, i)', old_expr='alpha*Vm')

    e_neu_model.model += 'x : integer (constant)\ny : integer (constant)\nz : integer (constant)\n'
    if precision == 'fp64':
        e_neu_model.modify_model('model',
                               'gtot = gtot0 + gtot1 + gtot2 + gtot3',
                               old_expr='gtot = gtot0')
        e_neu_model.model += 'gtot1 : volt\ngtot2 : volt\ngtot3 : volt\n'
        # changing original parameters to make simulations similar
        e_neu_model.modify_model('namespace', 150*pF, key='Cm')
    if precision == 'fp8':
        e_neu_model.modify_model('parameters',
                                 decimal2minifloat(.875),
                                 key='alpha_syn')

    # for emulating sleep
    # TODO timestamp and pattern with label to create_testbench? to be inserted into testbench
    # TODO sleep cycles (check ~/test.py)
    #sleep_time = events[sleep_iter][2]
    #wake_time = events[sleep_iter+1][1]
    #if precision == 'fp64':
    #    e_neu_model.modify_model(
    #        'model',
    #        'Iconst = 0*pA + 200*pA*int(t>sleep_time)*int(t<wake_time) : ampere',
    #        old_expr='Iconst : ampere')
    #else:
    #    e_neu_model.modify_model(
    #        'model',
    #        'Iconst = 0 + 200*int(t>sleep_time)*int(t<wake_time) : ampere',
    #        old_expr='Iconst : ampere')
    #del e_neu_model.parameters['Iconst']

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

    # Definition of readouts
    e_neu_model = LIFIP()

    # for emulating sleep
    #e_neu_model.model += 'dCa/dt = (500/501)*Ca/second : 1\n'
    #e_neu_model.reset += 'Ca += .1\n'

    e_neu_model.modify_model('parameters', 20*mV, key='Vthr')
    e_neu_model.modify_model('namespace', 0*mV, key='thr_inc')
    e_neu_model.modify_model('namespace', 120000*ms, key='tau_thr')

    # for when intrinsic plasticity is NOT to be used
    e_neu_model.modify_model(
        'model',
        'Vthr',
        old_expr='(alpha_thr*Vthr + dt*alpha_thr*thr_min/tau_thr)')

    # for when intrinsic plasticity is to be used
    #e_neu_model.modify_model('namespace', 0.1*mV, key='thr_inc')

    e_neu_model.modify_model('model', 'gtot = gtot0 + gtot1 + gtot2 + gtot3',
                             old_expr='gtot = gtot0')
    e_neu_model.model += 'gtot1 : volt\ngtot2 : volt\ngtot3 : volt\n'
    # TODO remove unused if hSTDP is used instead of normalization
    e_neu_model.model += 'inc_w : volt\n'
    e_neu_model.model += 'incoming_weights : volt\n'

    # for emulating sleep
    #e_neu_model.modify_model(
    #    'model',
    #    'Iconst = 0*pA + 200*pA*int(t>sleep_time)*int(t<wake_time) : ampere',
    #    old_expr='Iconst : ampere')
    #del e_neu_model.parameters['Iconst']

    readout = create_neurons(num_labels, e_neu_model, name='readout')

    teach_signal = SpikeGeneratorGroup(num_labels,
                                       [x[0] for x in events],
                                       [x[2] for x in events])

    # This for an artificial WTA
    #temp_time, temp_id = [], []
    #for ev in events:
    #    ev_interval = np.arange(ev[1]/ms, ev[2]/ms, 40)
    #    ev_label = [x for x in labels if x != ev[0]]
    #    aux_times = np.tile(ev_interval, len(ev_label))
    #    aux_ind = np.repeat(ev_label, len(ev_interval))
    #    temp_time.extend(aux_times)
    #    temp_id.extend(aux_ind)
    #antiteach_signal = SpikeGeneratorGroup(num_labels, temp_id, temp_time*ms)

    """ =================== Connections =================== """
    e_syn_model = liquid_syn()
    e_syn_model.modify_model('connection', .12, key='p')
    if precision == 'fp8':
        e_syn_model.modify_model('parameters',
                                 decimal2minifloat(96),
                                 key='weight')
    if precision == 'fp64':
        e_syn_model.modify_model('parameters', 80*mV, key='weight')
        e_syn_model.modify_model('model', 'gtot1_post', old_expr='gtot0_post')
        # changing original parameters to make simulations similar
        e_syn_model.modify_model('parameters', '7*ms', key='tau_syn')
    thl_conns = create_synapses(input_spikes, cells, e_syn_model)

    e_syn_model = liquid_syn()
    e_syn_model.modify_model(
        'connection',
        '.3 * exp(-((x_pre-x_post)**2 + (y_pre-y_post)**2 + (z_pre-z_post)**2) / 2**2)',
        key='p')
    e_syn_model.modify_model('parameters', '20*rand()*ms', key='delay')
    if precision == 'fp8':
        e_syn_model.modify_model('parameters',
                                 decimal2minifloat(24),
                                 key='weight')
    elif precision == 'fp64':
        e_syn_model.modify_model('model', 'gtot2_post', old_expr='gtot0_post')
        e_syn_model.modify_model('parameters', 20*mV, key='weight')
        # changing original parameters to make simulations similar
        e_syn_model.modify_model('parameters', '7*ms', key='tau_syn')
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
    if precision == 'fp8':
        i_syn_model.modify_model('namespace',
                                 decimal2minifloat(-1),
                                 key='w_factor')
        i_syn_model.modify_model('parameters',
                                 decimal2minifloat(120),
                                 key='weight')
    if precision == 'fp64':
        i_syn_model.modify_model('namespace', -1, key='w_factor')
        i_syn_model.modify_model('parameters', 100*mV, key='weight')
        i_syn_model.modify_model('model', 'gtot3_post', old_expr='gtot0_post')
        # changing original parameters to make simulations similar
        i_syn_model.modify_model('parameters', '7*ms', key='tau_syn')
    inh_inh = create_synapses(inh_cells, inh_cells, i_syn_model)

    i_syn_model.modify_model(
        'connection',
        '.4 * exp(-((x_pre-x_post)**2 + (y_pre-y_post)**2 + (z_pre-z_post)**2) / 2**2)',
        key='p')
    inh_exc = create_synapses(inh_cells, exc_cells, i_syn_model)

    e_syn_model = STDP()
    # TODO this stdp version could be used by default (wich also affects childs)?
    e_syn_model.modify_model('on_pre', 'i_trace += 1', old_expr='i_trace = 1')
    e_syn_model.modify_model('on_pre', '', old_expr='j_trace = 0')
    e_syn_model.modify_model('on_post', 'j_trace += 1', old_expr='j_trace = 1')
    e_syn_model.modify_model('on_post', '', old_expr='i_trace = 0')
    e_syn_model.modify_model('on_pre', 'g_syn += w_plast',
                             old_expr='g += w_plast')
    e_syn_model.modify_model('parameters', 0.02*mV, key='w_plast')

    # to test weight decay
    #asdf = 0.995
    #e_syn_model.modify_model('model', 
    #    'dw_plast/dt = asdf*w_plast/second : volt (clock-driven)',
    #    old_expr= 'w_plast : volt')
    #e_syn_model.modify_model('on_pre',
    #    '',
    #    old_expr= 'w_plast = clip(w_plast - eta*j_trace, 0*volt, w_max)')

    # only if hSTDP is used
    #e_syn_model.modify_model('namespace', 100*mV, key='w_lim')
    #e_syn_model.modify_model('model', '',
    #    old_expr='outgoing_weights_pre = w_plast : volt (summed)')
    #e_syn_model.modify_model('model', '',
    #    old_expr='outgoing_factor = outgoing_weights_pre - w_lim : volt')
    #e_syn_model.modify_model('model', '',
    #    old_expr='+ int(outgoing_factor > 0*volt)*outgoing_factor ')

    # TODO learning rate should be paired with normalization or hSTDP
    e_syn_model.modify_model('namespace', .01*mV, key='eta')

    # TODO do i need this? I DONT think so
    #e_syn_model.modify_model('parameters',
    #                         '20*rand()*ms',
    #                         key='delay')

    e_syn_model.model += 'inc_w_post = w_plast : volt (summed)\n'
    norm_factor = 1

    # needed for conventional normalization
    #e_syn_model.on_post += 'w_plast = int(norm_factor==1)*(w_plast/inc_w_post*mV) + int(norm_factor==0)*w_plast'

    e_syn_model.modify_model('model', 'dg_syn/dt = alpha_syn*g_syn',
                             old_expr='dg/dt = alpha_syn*g')
    e_syn_model.modify_model('model', 'g_syn*w_factor',
                             old_expr='g*w_factor')
    e_syn_model.modify_model('model', 'tau_syn_syn', old_expr='tau_syn')
    e_syn_model.parameters = {**e_syn_model.parameters,
                              **{'tau_syn_syn': '5*ms'}}
    e_syn_model.modify_model('model', 'alpha_syn_syn',
                             old_expr='alpha_syn')
    del e_syn_model.parameters['tau_syn']
    e_syn_model.parameters = {**e_syn_model.parameters,
                              **{'alpha_syn_syn': 'tau_syn_syn/(dt + tau_syn_syn)'}}
    del e_syn_model.parameters['alpha_syn']
    e_syn_model.modify_model('connection', .1, key='p')

    # for emulating sleep
    #e_syn_model.on_post += 'w_plast -= int(Iconst_post>0*pA)*int(Ca_post>1)*w_plast*.05\n'
    #e_syn_model.modify_model(
    #    'on_pre',
    #    'w_plast = clip(w_plast - int(Iconst_post==0*pA)*eta*j_trace, 0*volt, w_max)',
    #    old_expr='w_plast = clip(w_plast - eta*j_trace, 0*volt, w_max)')
    #e_syn_model.modify_model(
    #    'on_post',
    #    'w_plast = clip(w_plast + int(Iconst_post==0*pA)*eta*i_trace, 0*volt, w_max)\n',
    #    old_expr='w_plast = clip(w_plast + eta*i_trace, 0*volt, w_max)\n')

    e_syn_model.modify_model('parameters', '30*mV + 10*randn()*mV', key='w_plast')
    exc_readout = create_synapses(exc_cells, readout, e_syn_model,
                                  name='exc_readout')
    # only if hSTDP is used
    #exc_readout.run_regularly('w_plast = clip(w_plast - h_eta*heterosyn_factor, 0*volt, w_max)',
    #                           dt=1*ms)

    i_syn_model = CUBA()
    i_syn_model.modify_model('on_pre', 'g_syn += weight',
                             old_expr='g += weight')
    i_syn_model.modify_model('namespace', -1, key='w_factor')

    # if iSTDP is used instead
    #i_syn_model = iSTDP()
    #i_syn_model.modify_model('on_pre', 'i_trace += 1', old_expr='i_trace = 1')
    #i_syn_model.modify_model('on_pre', '', old_expr='j_trace = 0')
    #i_syn_model.modify_model('on_post', 'j_trace += 1', old_expr='j_trace = 1')
    #i_syn_model.modify_model('on_post', '', old_expr='i_trace = 0')
    #i_syn_model.modify_model('on_pre', 'g_syn += w_plast',
    #                         old_expr='g += w_plast')
    #i_syn_model.modify_model('namespace', 0.4, key='target_rate')
    #i_syn_model.modify_model('namespace', .01*mV, key='eta')

    i_syn_model.modify_model('model', 'gtot3_post = g_syn*w_factor',
                             old_expr='gtot0_post = g*w_factor')
    i_syn_model.modify_model('model', 'dg_syn/dt = alpha_syn*g_syn',
                             old_expr='dg/dt = alpha_syn*g')
    i_syn_model.modify_model('model', 'tau_syn_syn', old_expr='tau_syn')
    i_syn_model.parameters = {**i_syn_model.parameters,
                              **{'tau_syn_syn': '5*ms'}}
    i_syn_model.modify_model('model', 'alpha_syn_syn',
                             old_expr='alpha_syn')
    del i_syn_model.parameters['tau_syn']
    i_syn_model.parameters = {**i_syn_model.parameters,
                              **{'alpha_syn_syn': 'tau_syn_syn/(dt + tau_syn_syn)'}}
    del i_syn_model.parameters['alpha_syn']
    i_syn_model.modify_model('connection', .1, key='p')
    inh_readout = create_synapses(inh_cells, readout, i_syn_model,
                                  name='inh_readout')

    e_syn_model = CUBA()
    e_syn_model.modify_model('model', 'dg_syn/dt = alpha_syn*g_syn',
                             old_expr='dg/dt = alpha_syn*g')
    e_syn_model.modify_model('model', 'gtot1_post = g_syn*w_factor',
                             old_expr='gtot0_post = g*w_factor')
    e_syn_model.modify_model('on_pre', 'g_syn += weight',
                             old_expr='g += weight')
    e_syn_model.modify_model('connection', 'i', key='j')
    e_syn_model.modify_model('parameters', 50*mV, key='weight')
    label_readout = create_synapses(teach_signal, readout, e_syn_model,
                                    name='label_readout')

    # This for an artificial WTA
    #e_syn_model = CUBA()
    #e_syn_model.modify_model('model', 'dg_syn/dt = alpha_syn*g_syn',
    #                         old_expr='dg/dt = alpha_syn*g')
    #e_syn_model.modify_model('model', 'gtot2_post = g_syn*w_factor',
    #                         old_expr='gtot0_post = g*w_factor')
    #e_syn_model.modify_model('on_pre', 'g_syn += weight',
    #                         old_expr='g += weight')
    #e_syn_model.modify_model('connection', 'i', key='j')
    #e_syn_model.modify_model('parameters', -20*mV, key='weight')
    #antilabel_readout = create_synapses(antiteach_signal, readout, e_syn_model,
    #                                    name='antilabel_readout')

    """ =================== Results =================== """
    selected_exc_cells = np.random.choice(Ne, 4, replace=False)
    selected_inh_cells = np.random.choice(Ni, 4, replace=False)

    Metadata = {'selected_exc_cells': selected_exc_cells.tolist(),
                'selected_inh_cells': selected_inh_cells.tolist(),
                'dt': str(defaultclock.dt),
                'trial_no': trial_no,
                'duration': str(sim_dur)}
    with open(path+'metadata.json', 'w') as f:
        json.dump(Metadata, f)

    spkmon_e = SpikeMonitor(exc_cells)
    spkmon_i = SpikeMonitor(inh_cells)
    spkmon_ro = SpikeMonitor(readout)

    # This just for classify on input directly
    #spkmon_inp = SpikeMonitor(input_spikes)

    sttmon_e = StateMonitor(exc_cells, variables='Vm',
                            record=selected_exc_cells)
    sttmon_i = StateMonitor(inh_cells, variables='Vm',
                            record=selected_inh_cells)
    sttmon_ro = StateMonitor(readout, variables=['Vm', 'Iconst', 'Vthr'],
                             record=[x for x in range(num_labels)])

    kernel = kernels.GaussianKernel(sigma=30*q.ms)
    print('Running simulation')
    run(test_t)

    # TODO small weights e.g. 10 get stuck, high explodes e.g. 56. 1 kindda
    # works. 60 for small net
    # e_syn_model.modify_model('parameters', 60, key='weight')

    # this for just weight decay
    #readout.namespace['tau_thr'] = 30000*ms

    # if iSTDP is used
    #inh_readout.namespace['eta'] = 0*mV

    label_readout.namespace['w_factor'] = 0
    exc_readout.namespace['eta'] = 0*mV
    norm_factor = 0

    # just to test weight decay
    #asdf = 1

    # Adjusts weights to get response TODO maybe not needed anymore
    #exc_readout.w_plast = '145*(w_plast/inc_w_post*mV)'

    run(sim_dur-test_t)
    device.build()

    # TODO not used anymore. Idea was to count spikes instead of convolve by exp
    ## Process data for measuring accuracy
    #neo_spks = []
    #for spk_trains in spkmon_e.spike_trains().values():
    #    neo_spks.append(neo.SpikeTrain(spk_trains/ms*q.ms,
    #                                   t_stop=sim_dur/ms*q.ms))
    #data = BinnedSpikeTrain(neo_spks, bin_size=8*q.ms)
    #samples = []
    #for lt in events:
    #    # Not casting like below results in error!
    #    ti = np.around(lt[1]/ms).astype(int)*q.ms
    #    tf = np.around(lt[2]/ms).astype(int)*q.ms
    #    temp_data = data.time_slice(ti, tf).to_array()
    #    samples.append(temp_data.flatten())

    liquid_states = []
    sim_times = np.arange(0, sim_dur/ms+1, defaultclock.dt/ms)
    exp_kernel=[np.exp(-x/30) for x in range(1000)]

    ## spkmon_inp for classify on input directly
    #for spk_trains in list(spkmon_inp.spike_trains().values()):
    for spk_trains in (list(spkmon_e.spike_trains().values())
                       + list(spkmon_i.spike_trains().values())):
        conv_spks = np.zeros_like(sim_times)
        if len(spk_trains):
            conv_spks[np.around(spk_trains/defaultclock.dt).astype(int)] = 1
            conv_spks = np.convolve(conv_spks, exp_kernel)
        liquid_states.append(conv_spks[:len(sim_times)])

    # Linear classifier
    samples_size = len(events)
    train_size = samples_size-test_size
    lr = LinearSVC()
    liquid_states = np.array(liquid_states)
    labels_times = [int(lbls_ts[2]/defaultclock.dt) for lbls_ts in events]
    samples = liquid_states[:, labels_times]
    lr.fit(samples[:, :train_size].T, [x[0] for x in events][:train_size])
    acc = lr.score(samples[:, train_size:].T, [x[0] for x in events][train_size:])
    print(f'Accuracy was {acc}')
    with open(f'size_{size}-FP_{precision}-trial_{trial_no}.txt', 'w') as f:
        f.write(f'{acc:.2f}')

    temp_trains = spkmon_e.spike_trains()
    spk_trains = [neo.SpikeTrain(temp_trains[x]/ms, t_stop=sim_dur/ms, units='ms')
                  for x in temp_trains]
    pop_rates = statistics.instantaneous_rate(spk_trains,
                                              sampling_period=1*q.ms,
                                              kernel=kernel)
    pop_avg_rates = np.mean(pop_rates, axis=1)

    np.savez(f'{path}/exc_raster.npz',
             times=spkmon_e.t/ms,
             indices=spkmon_e.i)
    np.savez(f'{path}/inh_raster.npz',
             times=spkmon_i.t/ms,
             indices=spkmon_i.i)
    np.savez(f'{path}/rates.npz',
             times=np.array(pop_rates.times),
             rates=np.array(pop_avg_rates))

    # TODO remove it. Just wanted to compare membrane
    #test_time = sttmon_e.t/defaultclock.dt
    #if precision=='fp64': test_v = sttmon_e.Vm[0]/mV/20
    #elif precision=='fp8': test_v = minifloat2decimal(sttmon_e.Vm[0])/480
    #test_memb = pd.DataFrame({'time_ms': test_time, 'norm_Vm': test_v})
    #feather.write_dataframe(test_memb, f'{path}/test_memb.feather')

    if not quiet:
        fig,  (ax0, ax1, ax2, ax3) = plt.subplots(4, 1, sharex=True)
        for i in range(readout.N):
            plot_state(sttmon_ro.t, sttmon_ro.Vm[i], var_unit=mV, axes=ax0)
            plot_state(sttmon_ro.t, sttmon_ro.Vthr[i], var_unit=mV, axes=ax0)
        brian_plot(spkmon_ro, axes=ax3)
        ax0.vlines((test_t)/ms, 0, 1,
                   transform=ax0.get_xaxis_transform(), colors='r')
        brian_plot(spkmon_e, axes=ax1)
        ax2.plot(input_times/ms, input_indices, '.')

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax2.plot(pop_rates.times, pop_avg_rates, color='red')
        brian_plot(spkmon_e, marker=',', color='black', axes=ax1)
        ax1.set_xlabel(f'time ({pop_rates.times.dimensionality.latex})')
        ax1.set_ylabel('neuron number')
        ax2.set_ylabel(f'rate ({pop_rates.dimensionality})')

        plot_instantaneous_rates_colormesh(pop_rates)
        plt.title('Neuron rates on last trial')

        isi_neu = [isi(spks) for spks in spk_trains]
        fig, ax3 = plt.subplots()
        flatten_isi = []
        for vals in isi_neu:
            flatten_isi = np.append(flatten_isi, vals)
        ax3.hist(flatten_isi, bins=np.linspace(-3, 100, 10))
        ax3.set_title('ISI distribution')
        ax3.set_xlabel('ISI')
        ax3.set_ylabel('count')

        plt.figure()
        cv_neu = [cv(x) for x in isi_neu]
        plt.hist(cv_neu)
        plt.title('Coefficient of variation')
        plt.ylabel('count')
        plt.xlabel('CV')

        output_spikes = pd.DataFrame(
            {'time_ms': np.array(spkmon_ro.t/defaultclock.dt),
             'id': np.array(spkmon_ro.i)})
        feather.write_dataframe(output_spikes, f'{path}/output_spikes.feather')

        temp_time, temp_Vm, temp_Vthr, temp_id = [], [], [], []
        for idx in range(readout.N):
            temp_time.extend(sttmon_ro.t/defaultclock.dt)
            temp_Vm.extend(sttmon_ro.Vm[idx]/mV)
            temp_Vthr.extend(sttmon_ro.Vthr[idx]/mV)
            temp_id.extend([idx for _ in range(len(sttmon_ro.Vm[idx]))])
        output_traces = pd.DataFrame({'time_ms': temp_time,
                                      'Vm_mV': temp_Vm,
                                      'Vthr_mV': temp_Vthr,
                                      'id': temp_id})
        feather.write_dataframe(output_traces, f'{path}/output_traces.feather')

        input_spikes = pd.DataFrame(
            {'time_ms': input_times/defaultclock.dt,
             'id': input_indices})
        feather.write_dataframe(input_spikes, f'{path}/input_spikes.feather')

        rec_spikes = pd.DataFrame(
            {'time_ms': np.array(spkmon_e.t/defaultclock.dt),
             'id': np.array(spkmon_e.i)})
        feather.write_dataframe(rec_spikes, f'{path}/rec_spikes.feather')

        pd_events = np.array([[ev[0], ev[1]/defaultclock.dt, ev[2]/defaultclock.dt] for ev in events])
        pd_events = pd.DataFrame(pd_events, columns=['label', 'tstart_ms', 'tstop_ms'])
        feather.write_dataframe(pd_events, f'{path}/events_spikes.feather')

        links = pd.DataFrame(
            {'i': np.concatenate((exc_exc.i, exc_inh.i, inh_inh.i+Ne, inh_exc.i+Ne)),
             'j': np.concatenate((exc_exc.j, exc_inh.j+Ne, inh_inh.j+Ne, inh_exc.j))
             })
        feather.write_dataframe(links, f'{path}/links.feather')
        nodes = pd.DataFrame(
            {'neu_id': [x for x in range(Nt)],
             'type': ['exc' for _ in range(Ne)] + ['inh' for _ in range(Ne, Nt)]})
        feather.write_dataframe(nodes, f'{path}/nodes.feather')

        # for emulating sleep
        #plt.figure()
        #plt.plot(sttmon_ro.Ca[0])
        #plt.figure()
        #plt.plot(sttmon_ro.Ca[1])

        plt.show()
