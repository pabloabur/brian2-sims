from brian2 import *
from lfsr import create_lfsr

defaultclock.dt = 1*ms
prefs.codegen.target = "numpy"

# Generate lfsr numbers
G = NeuronGroup(4, '''v = decay_prob : volt
                      decay_prob = ta( ((seed+t) % lfsr_max_value) + lfsr_init ) / (2**lfsr_num_bits-1) : volt
                      lfsr_max_value : second
                      lfsr_num_bits : 1
                      seed : second
                      lfsr_init : second''')
# Set parameter
G.lfsr_num_bits = 3

lfsr = create_lfsr(G.lfsr_num_bits)

G.lfsr_max_value = lfsr['max_value']*ms
G.lfsr_init = lfsr['init']*ms
G.seed = lfsr['seed']*ms

ta = TimedArray(lfsr['array']*mV, dt=defaultclock.dt)
mon = StateMonitor(G, 'v', record=True)
net = Network(G, mon)
net.run(7*ms)
print(mon.v[0])
print(mon.v[1])
print(mon.v[2])
print(mon.v[3])
