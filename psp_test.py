""" Peak around 0.14mV, rise time and half-width around 1.7ms and 8.5ms respectively"""
""" We only use one state variable to describe current, which is not enough to replicate
    the original work. Just for that reason, we need to change the parameters a bit."""
from brian2 import *
from scipy.integrate import solve_ivp

dt = 0.1
defaultclock.dt = dt * ms

F = lambda t, v: -v/10 + Is(t)/250
def Is(t): return 45.61*exp(1)/0.33*t*exp(-t/0.33)

t_eval = arange(0, 20, dt)
sol = solve_ivp(F, [0, 20], [0], t_eval=t_eval)

# TODO
tau_s = .66 # was .33
weight = 25 # was 45.61

eqs = f'''
dV/dt = (40*Mohm*I-V)/(10*ms) : volt
dI/dt = (-I + x)/({tau_s}*ms) : ampere
dx/dt = -x/({tau_s}*ms) : ampere
'''
N = 2
G = NeuronGroup(N, eqs, threshold='V>20*mV', reset='V=0*mV', method='euler')

# TODO mimic original work like GeNN or our approximation (e multiplied for each)
#S = Synapses(G, G, 'weight : ampere', on_pre=f'x += ({weight}*pA*exp(1))')
#S = Synapses(G, G, 'weight : ampere', on_pre='x += (45.61*pA)')
S = Synapses(G, G, 'weight : ampere', on_pre=f'I += ({weight}*pA*exp(1))')
#S = Synapses(G, G, 'weight : ampere', on_pre='I += (45.61*pA)')
S.connect(i=0, j=1)
G.V = [30, 0]*mV
mon = StateMonitor(G, variables=['V', 'I'], record=True)
run(20*ms)

figure()
plot(mon.t/ms, mon.I[1]/pA, label='handwritten I(t)')
plot(mon.t/ms, [Is(t) for t in mon.t/ms], label='Our I(t)')
ylabel('pA')
xlabel('ms')
legend()
figure()
plot(mon.t/ms, mon.V[1]/mV, label='handwritten V(t)')
plot(sol.t, sol.y[0], label='Our Vm')
ylabel('mV')
xlabel('ms')
legend()
show()
