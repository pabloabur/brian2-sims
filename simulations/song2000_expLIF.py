from brian2 import *
from teili.models.neuron_models import ExpLIF as teili_neu
from teili.models.synapse_models import ExponentialStdp as teili_syn
from teili.core.groups import Neurons, Connections

N = 1000
F = 15*Hz
input = PoissonGroup(N, rates=F)

neurons = Neurons(1, equation_builder=teili_neu(num_inputs=1))
#taum = 10*ms
neurons.Cm = 140*pF # 281
neurons.gL = 14*nS # 4.3
neurons.EL = -74*mV # -55
neurons.Vres = -60*mV # -70.6
neurons.VT = -54*mV # -50.4
neurons.DeltaT = 0.001*mV # -2
neurons.refP = 0*ms # 2
#Ee = 0*mV N.A. I_syn already does it.

S = Connections(input, neurons, equation_builder=teili_syn)
S.connect()
S.tausyn = 5*ms
S.taupre = 20*ms # 10
S.taupost = S.taupre
S.w_max = .01 # 1
S.dApre = .01 # 0.1
S.Q_diffAPrePost = 1.05
S.w_plast = 'rand() * w_max'
init_w_plast = array(S.w_plast)
S.weight = 1

rec_w = choice(N, 5, replace=False)
mon = StateMonitor(S, ['w_plast', 'Apre', 'Apost'], record=rec_w)
s_mon = SpikeMonitor(input)
r_mon = PopulationRateMonitor(neurons)

run(100*second, report='text')

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
