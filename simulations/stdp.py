""" This file has modified versions of some code from Teili. See
M. Milde, A. Renner, R. Krause, A. M. Whatley, S. Solinas, D. Zendrikov,
N. Risi, M. Rasetto, K. Burelo, V. R. C. Leite. teili: A toolbox for building
and testing neural algorithms and computational primitives using spiking
neurons. Unreleased software, Institute of Neuroinformatics, University of
Zurich and ETH Zurich, 2018.
"""

import numpy as np
import feather
import pandas as pd

from brian2 import run, device, second, ms, defaultclock, SpikeGeneratorGroup,\
    StateMonitor, SpikeMonitor, TimedArray

from brian2tools import brian_plot, plot_state
import matplotlib.pyplot as plt

from core.utils.misc import minifloat2decimal, decimal2minifloat

from core.equations.neurons.fp8LIF import fp8LIF
from core.equations.synapses.fp8CUBA import fp8CUBA
from core.equations.synapses.fp8STDP import fp8STDP
from core.builder.groups_builder import create_synapses, create_neurons


def stimuli2():
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
    return tapre, tapost, tmax, N, trial_duration

def stimuli(isi=10):
    """Stimulus gneration for STDP protocols.

    This function returns two brian2 objects.
    Both are Spikegeneratorgroups which hold a single index each
    and varying spike times.
    The protocol follows homoeostasis, weak LTP, weak LTD, strong LTP,
    strong LTD, homoeostasis.

    Args:
        isi (int, optional): Interspike Interval. How many spikes per stimulus
            phase.

    Returns:
        SpikeGeneratorGroup (brian2.obj: Brian2 objects which hold the
            spiketimes and the respective neuron indices.
    """
    t_pre_homoeotasis_1 = np.arange(3, 304, isi)
    t_pre_weakLTP = np.arange(403, 604, isi)
    t_pre_weakLTD = np.arange(703, 904, isi)
    t_pre_strongLTP = np.arange(1003, 1204, isi)
    t_pre_strongLTD = np.arange(1303, 1504, isi)
    t_pre_homoeotasis_2 = np.arange(1603, 1804, isi)
    t_pre = np.hstack((t_pre_homoeotasis_1, t_pre_weakLTP, t_pre_weakLTD,
                       t_pre_strongLTP, t_pre_strongLTD, t_pre_homoeotasis_2))

    # Normal distributed shift of spike times to ensure homoeotasis
    t_post_homoeotasis_1 = t_pre_homoeotasis_1 + \
        np.random.randint(-3, 3, len(t_pre_homoeotasis_1))
    t_post_weakLTP = t_pre_weakLTP + 5   # post neuron spikes 7 ms after pre
    t_post_weakLTD = t_pre_weakLTD - 5   # post neuron spikes 7 ms before pre
    t_post_strongLTP = t_pre_strongLTP + 1  # post neurons spike 1 ms after pre
    t_post_strongLTD = t_pre_strongLTD - 1  # and 1 ms before pre
    t_post_homoeotasis_2 = t_pre_homoeotasis_2 + \
        np.random.randint(-3, 3, len(t_pre_homoeotasis_2))

    t_post = np.hstack((t_post_homoeotasis_1, t_post_weakLTP, t_post_weakLTD,
                        t_post_strongLTP, t_post_strongLTD, t_post_homoeotasis_2
                        ))
    ind_pre = np.zeros(len(t_pre))
    ind_post = np.zeros(len(t_post))

    pre = SpikeGeneratorGroup(
        1, indices=ind_pre, times=t_pre * ms, name='gPre')
    post = SpikeGeneratorGroup(
        1, indices=ind_post, times=t_post * ms, name='gPost')

    return pre, post


def stdp(args):
    defaultclock.dt = args.timestep * ms

    if args.protocol == 1:
        N_pre, N_post = 2, 2
        pre_spikegenerator, post_spikegenerator = stimuli(isi=30)
        neuron_model = fp8LIF()
        pre_neurons = create_neurons(2, neuron_model)
        post_neurons = create_neurons(2, neuron_model)

        synapse_model = fp8CUBA()
        synapse_model.modify_model('parameters',
                                   decimal2minifloat(192),
                                   key='weight')
        pre_synapse = create_synapses(pre_spikegenerator,
                                      pre_neurons,
                                      synapse_model)
        post_synapse = create_synapses(post_spikegenerator,
                                       post_neurons,
                                       synapse_model)

        tmax = 2. * second

    elif args.protocol == 2:
        tapre, tapost, tmax, N, trial_duration = stimuli2()
        N_pre, N_post = N, N
        neuron_model = fp8LIF()
        neuron_model.modify_model(
            'model',
            'summed_decay = tapre(t, i)',
            old_expr='summed_decay = fp8_add(decay_term, gtot*int(not_refractory))')
        neuron_model.modify_model('threshold', '1', old_expr='Vthr')
        # TODO why is it not working with Ca_inc=50 (i.e. 0.625_10)??
        # It decays so becomes too small. Find one a bit bigger
        neuron_model.modify_model('namespace', 127, key='Ca_inc')
        # TODO do I need tmax? and why not on run?
        neuron_model.namespace = {**neuron_model.namespace,
                                  'tmax': tmax,
                                  'tapre': tapre}
        pre_neurons = create_neurons(N, neuron_model)

        neuron_model = fp8LIF()
        neuron_model.modify_model(
            'model',
            'summed_decay = tapost(t, i)',
            old_expr='summed_decay = fp8_add(decay_term, gtot*int(not_refractory))')
        neuron_model.modify_model('threshold', '1', old_expr='Vthr')
        neuron_model.modify_model('namespace', 127, key='Ca_inc')
        neuron_model.namespace = {**neuron_model.namespace,
                                  'tmax': tmax,
                                  'tapost': tapost}
        post_neurons = create_neurons(N, neuron_model)

        tmax = tmax * ms

    elif args.protocol == 3:
        from brian2 import Hz, PoissonGroup
        N_pre = 1000
        N_post = 1
        tmax = 1000 * ms # TODO was 150s??
        # TODO below instead? does not work...
        # neuron_model = fp8LIF()
        # neuron_model.model += '\nrates : Hz'
        # neuron_model.modify_model('threshold',
                                  # 'rand()<rates*dt',
                                  # old_expr='Vm == Vthr')
        # pre_neurons = create_neurons(N_pre, neuron_model)
        # stim = TimedArray(np.reshape([15 for _ in range(int(N_pre*(tmax/ms)))], (int(tmax/ms), N_pre))*Hz, dt=defaultclock.dt)
        # pre_neurons.rates = 'stim(t, i)' #15*Hz
        pre_neurons = PoissonGroup(N_pre, 15*Hz)

        # TODO random init weight

        neuron_model = fp8LIF()
        post_neurons = create_neurons(N_post, neuron_model)

    if args.protocol == 1 or args.protocol == 2:
        conn_condition = 'i==j'
    if args.protocol == 3:
        # None makes it all to all
        conn_condition = None

    stdp_model = fp8STDP()
    stdp_model.modify_model('connection',
                            conn_condition,
                            key='condition')
    stdp_model.modify_model('parameters',
                            decimal2minifloat(0.001953125),
                            key='w_plast')
    if args.protocol == 3:
        import pdb;pdb.set_trace()
        stdp_model.model += ('dCa_syn/dt = fp8_multiply(Ca_syn, 55)/second : 1\n'
                             + 'spiked : second\n')
        stdp_model.on_pre += ('Ca_syn = fp8_add(Ca_syn, 50)\n'
                              + 'spiked = t\n')
        stdp_model.modify_model('on_post', 'Ca_syn', old_expr='Ca_pre')
    stdp_synapse = create_synapses(pre_neurons, post_neurons, stdp_model)

    # Setting up monitors
    spikemon_pre_neurons = SpikeMonitor(pre_neurons,
                                        name='spikemon_pre_neurons')
    spikemon_post_neurons = SpikeMonitor(post_neurons,
                                         name='spikemon_post_neurons')
    # statemon_pre_neurons = StateMonitor(pre_neurons,
    #                                     variables=['Vm', 'g', 'Ca'],
    #                                     record=range(N_pre),
    #                                     name='statemon_pre_neurons')
    statemon_post_neurons = StateMonitor(post_neurons,
                                         variables=['Vm', 'g', 'Ca'],
                                         record=range(N_post),
                                         name='statemon_post_neurons')
    statemon_post_synapse = StateMonitor(stdp_synapse,
                                         variables=['w_plast'],
                                         record=range(N_post),
                                         name='statemon_post_synapse')

    run(tmax)
    #average_wplast[avg_trial, :] = np.array(stdp_synapse.w_plast)

    if args.backend == 'cpp_standalone':
        device.build(args.code_path)

    if not args.quiet:
        if args.protocol == 1:
            brian_plot(spikemon_pre_neurons)
            plt.savefig(f'{args.save_path}/fig1')

            plt.figure()
            brian_plot(spikemon_post_neurons)
            plt.savefig(f'{args.save_path}/fig2')

            plt.figure()
            plot_state(statemon_post_synapse.t,
                       minifloat2decimal(statemon_post_synapse.w_plast[0]),
                       var_name='weight')
            plt.savefig(f'{args.save_path}/fig3')

            plt.figure()
            plot_state(statemon_post_neurons.t,
                       minifloat2decimal(statemon_post_neurons.Ca[0]),
                       var_name='Time window')
            plt.savefig(f'{args.save_path}/fig4')

            plt.figure()
            plot_state(statemon_pre_neurons.t,
                       minifloat2decimal(statemon_pre_neurons.Vm[0]),
                       var_name='Vm')
            plt.savefig(f'{args.save_path}/fig5')
        # TODO why is one of them huge and other not? Apparently one is not 
        # getting enough excitation
        elif args.protocol == 2:
            labels = ('indices', 'times', 'type')
            data = [(np.array(spikemon_pre_neurons.i),
                     spikemon_pre_neurons.t/ms,
                     ['pre' for _ in range(len(spikemon_pre_neurons.t))]),
                    (np.array(spikemon_post_neurons.i),
                     spikemon_post_neurons.t/ms,
                     ['post' for _ in range(len(spikemon_post_neurons.t))])
                    ]
            temp_data = {l: [] for l in labels}
            for dat in data:
                for idx in range(len(dat)):
                    temp_data[labels[idx]].extend(dat[idx])
            spikes = pd.DataFrame(temp_data)
            feather.write_dataframe(spikes, args.save_path + 'spikes.feather')
            import pdb;pdb.set_trace()

            # TODO each timing is associate with one synase? In that case I just
            # redefine init_wplast to subtract it and plot over pairs_timing
            pairs_timing = (spikemon_post_neurons.t[:trial_duration]
                            - spikemon_post_neurons.t[:trial_duration][::-1])/ms
            import plotext as pt
            pt.plot(pairs_timing, minifloat2decimal(statemon_post_synapse.w_plast[:, -1]))
            pt.show()

            pt.clear_figure()
            pt.plot(statemon_pre_neurons.Ca[0][:100])
            pt.plot(statemon_post_neurons.Ca[0][:100])
            pt.show()

            # TODO we don't need averages anymore; do it outside
            # TODO average_wplast[avg_trial-1, :]-init_wplast #  one sample (in case of average)
            # np.mean(average_wplast, axis=0)-init_wplast #  average (in case of average)
            # TODO Apre's and Apost's, weak and strong. Maybe not necessary
        elif args.protocol == 3:
            # TODO adapt below
            pdb.set_trace()
            plt.subplot(311)
            plt.hist(S.w_plast / S.w_max, 20)
            plt.xlabel('Weight / w_max')
            plt.subplot(312)
            plt.plot(mon.t/second, mon.w_plast.T/S.w_max[0])
            plt.xlabel('Time (s)')
            plt.ylabel('Weight / w_max')
            plt.subplot(313)
            plt.plot(neu_mon.t/ms/1000, neu_mon.i, '.k')
            plt.xlabel('Time (s)')
            plt.ylabel('spikes')
