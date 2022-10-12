""" This is a standard, full-precision LIF neuron model. Equation obtained
    using backward Euler (dt=1*ms) and must be used with a custom integration
    method.
"""

from brian2.units import *
from core.equations.base_equation import BaseNeuron, ParamDict

class LIF(BaseNeuron):
    def __init__(self):
        super().__init__()
        self.model = '''
            dVm/dt = (alpha*Vm + dt*alpha*(gtot*gl + Iconst)/Cm)/second : volt (unless refractory)

            alpha : 1 (constant)
            tau_m : second (constant)
            gtot = gtot0 : volt
            gtot0 : volt
            Iconst : ampere
            '''
        self.threshold = 'Vm > Vthr'
        self.refractory = '2*ms'
        self.reset = '''
            Vm = Vreset
            '''
        self.namespace = ParamDict({
            'Cm': 200*pF,
            'gl': 10*nS,
            'Vreset': 0*mV,
            'Vthr': 20*mV
            })
        self.parameters = ParamDict({
            'tau_m': 'Cm/gl',
            'Vm': 'Vreset',
            'Iconst': '0*pA',
            'alpha': 'tau_m/(dt + tau_m)'
            })
