""" This is a standard, full-precision LIF neuron model. Equation obtained
    using backward Euler (dt=1*ms) and must be used with a custom integration
    method.
"""

from brian2.units import *
from core.equations.base_equation import BaseNeuron, ParamDict


class tsvLIF(BaseNeuron):
    def __init__(self):
        super().__init__()
        self.model = '''
            dVm/dt = (alpha*Vm + dt*alpha*(g*gl + Iconst)/Cm)/second : volt (unless refractory)
            dg/dt = alpha_syn*g/second : volt
            dCa/dt = alpha_ca*Ca/second : 1

            tau_m : second (constant)
            alpha : 1 (constant)
            tau_syn : second (constant)
            alpha_syn : 1 (constant)
            tau_ca : second (constant)
            alpha_ca : 1 (constant)
            Iconst : ampere
            '''
        self.threshold = 'Vm > Vthr'
        self.refractory = '2*ms'
        self.reset = '''
            Vm = Vreset
            Ca += Ca_inc
            Ca *= -1
            '''
        self.events = ParamDict({'active_Ca': 'abs(Ca) > 0'})
        self.namespace = ParamDict({
            'Cm': 200*pF,
            'gl': 10*nS,
            'Vreset': 0*mV,
            'Vthr': 20*mV,
            'Ca_inc': 1
            })
        self.parameters = ParamDict({
            'Vm': 'Vreset',
            'Ca': '0',
            'g': '0*mV',
            'Iconst': '0*pA',
            'tau_m': 'Cm/gl',
            'alpha': 'tau_m/(dt + tau_m)',
            'tau_syn': '5*ms',
            'alpha_syn': 'tau_syn/(dt + tau_syn)',
            'tau_ca': '20*ms',
            'alpha_ca': 'tau_ca/(dt + tau_ca)',
            })
