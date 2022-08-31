from brian2 import *

N = 1000
taum = 10*ms
taupre = 20*ms
taupost = taupre
Ee = 0*mV
vt = -54*mV
vr = -60*mV
El = -74*mV
taue = 5*ms
F = 15*Hz
w_max = .01
dApre = .01
dApost = -dApre * taupre / taupost * 1.05
dApost *= w_max
dApre *= w_max

eqs_neurons = '''
dv/dt = (ge * (Ee-v) + El - v) / taum : volt
dge/dt = -ge / taue : 1
'''

input = PoissonGroup(N, rates=F)
neurons = NeuronGroup(1, eqs_neurons, threshold='v>vt', reset='v = vr',
                              method='euler')
S = Synapses(input, neurons,
                     '''w_plast : 1
                        dApre/dt = -Apre / taupre : 1 (event-driven)
                        dApost/dt = -Apost / taupost : 1 (event-driven)''',
                        on_pre='''ge += w_plast
                        Apre += dApre
                        w_plast = clip(w_plast + Apost, 0, w_max)''',
                        on_post='''Apost += dApost
                        w_plast = clip(w_plast + Apre, 0, w_max)''',
                        )
S.connect()
S.w_plast = 'rand() * w_max'
init_w_plast = array(S.w_plast)
rec_w = choice(N, 5, replace=False)
mon = StateMonitor(S, ['w_plast', 'Apre', 'Apost'], record=rec_w)
s_mon = SpikeMonitor(input)
r_mon = PopulationRateMonitor(neurons)

run(100*second, report='text')

subplot(321)
plot(S.w_plast / w_max, '.k')
ylabel('Weight / w_max')
xlabel('Synapse index')
subplot(322)
hist(init_w_plast / w_max, 20, alpha=0.4, color='r', label='Before')
hist(S.w_plast / w_max, 20, alpha=0.4, color='b', label='After')
xlabel('Weight / w_max')
legend()
ax1 = subplot(323)
for idx, s_idx in enumerate(rec_w):
    plot(mon.t/second, mon.w_plast[idx].T/w_max, label=f'Syn. {s_idx}')
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
