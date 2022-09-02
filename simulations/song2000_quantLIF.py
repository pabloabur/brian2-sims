from brian2 import *
from teili.models.neuron_models import QuantStochLIF as teili_neu
from teili.models.synapse_models import QuantStochSynStdp as teili_syn
from teili.core.groups import Neurons, Connections
from teili.tools.misc import neuron_group_from_spikes

# You could also try no Iconst, 1/32 gain_syn, extra plasticity mechanisms
# and heterogeneous taus
from teili.models.builder.synapse_equation_builder import SynapseEquationBuilder
teili_syn = SynapseEquationBuilder(base_unit='quantized',
          plasticity='quantized_stochastic_stdp')#,
          #pairing = 'stochastic_reduced_symmetric',
          #structural_plasticity='stochastic_counter',
          #compensatory_process = 'stochastic_heterosynaptic',
          #prob_release = 'stochastic_syn_release')

defaultclock.dt = 1 * ms

sim_duration = 100*second
N = 1000
F = 15*Hz
input = PoissonGroup(N, rates=F)
temp_monitor = SpikeMonitor(input, name='temp_monitor')
run(sim_duration, report='text')
input = neuron_group_from_spikes(
    N, defaultclock.dt, sim_duration,
    spike_indices=np.array(temp_monitor.i),
    spike_times=np.array(temp_monitor.t)*second)
del temp_monitor

neurons = Neurons(1, method=ExplicitStateUpdater('''x_new = f(x,t)'''),
    equation_builder=teili_neu(num_inputs=1))
neurons.tau = 10*ms
neurons.Vrest = -12*mV
neurons.Vreset = 0*mV
neurons.Vthr = 15*mV
neurons.refP = 0*ms
neurons.Vm_min = -12
neurons.Vm_max = 15
neurons.refrac_tau = 0*ms
neurons.Iconst = 16.5*mA

S = Connections(input, neurons, method=ExplicitStateUpdater('''x_new = f(x,t)'''),
    equation_builder=teili_syn)
S.connect()
S.tausyn = 5*ms
random_array = .5*randn(N)
random_array = clip(1 + random_array, 0.5, 3)
S.taupre = 20*ms*random_array
S.taupost = 20*ms
S.w_max = 15
S.dApre = 15
S.rand_num_bits_Apre = 4
S.rand_num_bits_Apost = 4
S.stdp_thres = 1
S.weight = 1
S.gain_syn = (1/64)*mA
S.w_plast = 'rand() * w_max'
init_w_plast = array(S.w_plast)

rec_w = choice(N, 5, replace=False)
mon = StateMonitor(S, ['w_plast', 'Apre', 'Apost'], record=rec_w)
s_mon = SpikeMonitor(input)
r_mon = PopulationRateMonitor(neurons)

run(sim_duration, report='text')

subplot(321)
plot(S.w_plast / S.w_max[0], '.k')
ylabel('Weight / w_max')
xlabel('Synapse index')
subplot(322)
hist(init_w_plast / S.w_max[0], 20, alpha=0.4, color='r', label='Before')
hist(S.w_plast / S.w_max[0], 20, alpha=0.4, color='b', label='After')
xlabel('Weight / w_max')
legend()
ax1 = subplot(323)
for idx, s_idx in enumerate(rec_w):
    plot(mon.t/second, mon.w_plast[idx].T/S.w_max[0], label=f'Syn. {s_idx}')
xlabel('Time (s)')
ylabel('Weight / w_max')
ylim(0, 1)
legend()
subplot(324, sharex=ax1)
plot(r_mon.t/second, r_mon.smooth_rate(width=50*ms)/Hz)
xlabel('Time (s)')
ylabel('Rate (Hz)')
subplot(325, sharex=ax1)
for idx, s_idx in enumerate(rec_w):
    plot(mon.t/second, mon.Apre[idx])
xlabel('Time (s)')
ylabel('Pre time window')
subplot(326, sharex=ax1)
for idx, s_idx in enumerate(rec_w):
    plot(mon.t/second, mon.Apost[idx])
xlabel('Time (s)')
ylabel('Post time window')
tight_layout()
show()
