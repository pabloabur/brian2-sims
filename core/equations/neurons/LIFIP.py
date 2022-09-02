""" This is a standard, full-precision LIF neuron model with Intrinsic
    Plasticity, obtained by decaying threshold. Equation obtained using
    backward Euler and must be used with a custom integration method.
"""

from brian2.units import *
from core.equations.base_equation import ParamDict
from core.equations.neurons.LIF import LIF

class LIFIP(LIF):
    def __init__(self):
        super().__init__()
        self.model += ('dVthr/dt = (alpha_thr*Vthr + dt*alpha_thr*thr_min/tau_thr)/second : volt\n'
                        + 'alpha_thr : 1 (constant)\n')
        self.reset += 'Vthr = clip(Vthr+thr_inc, thr_min, thr_max)\n'
        self.namespace = ParamDict({**self.namespace, **{'thr_min': 1*mV,
                                               'thr_max': 40*mV,
                                               'tau_thr': 30000*ms,
                                               'thr_inc': 0.01*mV}})
        self.parameters = ParamDict({**self.parameters, **{'Vthr': 10*mV,
                                                  'alpha_thr' : 'tau_thr/(dt + tau_thr)',
                                                  'Vm': 'Vreset'}})

        del self.namespace['Vthr']
