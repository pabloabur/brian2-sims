from core.utils.misc import fp8_add, fp8_multiply, minifloat2decimal, decimal2minifloat

import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

N = 3
T = 24
eta = 50


class neuron:
    def __init__(self):
        self.vm = 0
        self.psc = 0
        self.tw = 0
        self.active_spike = 0

    def leak_vm(self):
        if self.vm <= 127:
            self.vm = fp8_multiply(self.vm, 55, 0)
        else:
            self.vm = fp8_multiply(self.vm, 1, 0)

    def leak_psc(self):
        self.psc = fp8_multiply(self.psc, 53, 0)

    def leak_tw(self):
        self.tw = fp8_multiply(self.tw, 55, 0)
        if self.active_spike:
            self.active_spike = 0

    def integrate_current(self):
        if self.vm <= 127:
            self.vm = fp8_add(self.vm, self.psc, 0)

    def threshold(self):
        if self.vm == 127:
            self.vm = 177
            self.tw = 127
            self.active_spike = 1


def time_driven_module(neurons):
    # TODO for active
    # TODO in brian, this is the 'model', 'threshold', and 'reset'
    for n in neurons:
        n.leak_vm()
        n.leak_psc()
        n.leak_tw()
        n.integrate_current()
        n.threshold()


def event_driven_module(neurons):
    # TODO in brian triggered for each neuron if 'threshold' found spike. We
    # do it for all active (set in memory somewhere), also checking state
    # variables. We could do nothing in 'on_post' and 'on_pre' and do our loop
    # with a run_reg on syapses that have access to proper namespace and
    # when='after_resets'
    for pre_id, n in enumerate(neurons):
        # TODO crossbar-like memory access, i.e. M*N or rho*M*N
        fan_out_idx = [i for i, x in enumerate(fan_out[pre_id]) if x == 1]

        # TODO relates to sparsity of the network, i.e. propagates
        # TODO for active TW
        if n.active_spike == 1:
            for post_id in fan_out_idx:
                neurons[post_id].psc = weights[pre_id][post_id]

                if (neurons[post_id].active_spike == 0
                        and neurons[post_id].tw > 0):
                    delta_w = fp8_multiply(neurons[post_id].tw, eta, 0)
                    delta_w = fp8_multiply(delta_w, 184, 0)  # multiply by -1
                    weights[pre_id][post_id] = fp8_add(weights[pre_id][post_id],
                                                       delta_w,
                                                       0)

        # TODO Strategy to make updates on post-event only
        elif n.tw > 0:
            for post_id in fan_out_idx:
                if neurons[post_id].active_spike == 1:
                    delta_w = fp8_multiply(neurons[pre_id].tw, eta, 0)
                    weights[pre_id][post_id] = fp8_add(weights[pre_id][post_id],
                                                       delta_w,
                                                       0)


neurons = [neuron() for _ in range(N)]
# post:     A  B  C     /pre:
fan_out = [[0, 1, 1],  # A
           [0, 0, 0],  # B
           [0, 1, 0]]  # C
# post:     A  B  C     /pre:
weights = [[0, 115, 120],  # A
           [0,   0,   0],  # B
           [0, 115,   0]]  # C
ref_weights = deepcopy(weights)
# TODO we expect: - A->C increase
#                 - A->B/C->B increase
#                 - A->C/->B smaller decrease

vm_monitor = np.zeros((N, 1))
psc_monitor = np.zeros((N, 1))
tw_monitor = np.zeros((N, 1))

neurons[0].psc = 120
neurons[1].psc = 180
neurons[2].psc = 130

# TODO active matrix: 1 if any "compartments" is non-zero

for t in range(T):
    if t == 20:
        neurons[0].psc = 115
    time_driven_module(neurons)
    E = 100
    for e in range(E):
        event_driven_module(neurons)

    record_vm = np.array([[n.vm for n in neurons]])
    vm_monitor = np.concatenate((vm_monitor, record_vm.T), axis=1)
    record_psc = np.array([[neurons[i].psc for i in range(N)]])
    psc_monitor = np.concatenate((psc_monitor, record_psc.T), axis=1)
    record_tw = np.array([[neurons[i].tw for i in range(N)]])
    tw_monitor = np.concatenate((tw_monitor, record_tw.T), axis=1)

f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
ax1.plot(minifloat2decimal(tw_monitor[0]))
ax1.set_title('Neuron A')

ax2.plot(minifloat2decimal(tw_monitor[1]))
ax2.set_title('Neuron B')

ax3.plot(minifloat2decimal(tw_monitor[2]))
ax3.set_title('Neuron C')

plt.tight_layout()
plt.show()
