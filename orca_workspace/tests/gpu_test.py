from brian2 import *
from timeit import timeit
#import brian2genn

#set_device('genn')
stochastic_decay = ExplicitStateUpdater('''x_new = f(x,t)''')
defaultclock.dt = 1*ms

@implementation('cpp', '''
     double piecewise_linear(double I) {
        if (I < 1e-9)
            return 0;
        if (I > 100)
            return 100;
        return I;
     }
     ''')
@check_units(I=1, result=1)
def piecewise_linear(I):
    return clip(I, 0, 100)

n = 100
duration = 200*second
tau = 10*ms
eqs = '''
dv/dt = tau/(tau+dt)*v/second : volt (unless refractory)
I = piecewise_linear(-50) : 1
'''
group = NeuronGroup(n, eqs, threshold='v > 10*mV', reset='v = 0*mV',
                    refractory=5*ms, method=stochastic_decay)
group.v = 0*mV

S1 = Synapses(group, group, on_pre='v_post += 1*mV', method=stochastic_decay)
S1.connect(p=0.05)

P = PoissonGroup(100, rates=10*Hz)
S2 = Synapses(P, group, on_pre='v_post += 1*mV', method=stochastic_decay)
S2.connect(p=0.6)

monitor = SpikeMonitor(group)
mon2 = StateMonitor(group, variables=['v'], record=True)

def test():
    run(duration)

print(timeit("test()", globals=globals(), number=1))
