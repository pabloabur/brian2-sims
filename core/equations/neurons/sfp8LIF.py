""" Model with 8-bit floating point resolution"""

from brian2.units import *
from core.equations.base_equation import BaseNeuron, ParamDict


class sfp8LIF(BaseNeuron):
    def __init__(self):
        super().__init__()
        self.model = '''
            dVm/dt = int(forbidden_cond == 0) * summed_decay / second : 1
            # Prevents negative current to bring voltage below rest
            forbidden_cond = fp8_smaller_than(summed_decay, Vrest) * int(not_refractory) : integer
            summed_decay = fp8_add_stochastic(decay_term, gtot*int(not_refractory)) : integer
            decay_term = fp8_multiply_stochastic(Vm, alpha)*int(not_refractory) + fp8_multiply_stochastic(Vm, alpha_refrac)*int(not not_refractory) : integer
            gtot = fp8_add_stochastic(g, Iconst) : integer

            dg/dt = fp8_multiply_stochastic(g, alpha_syn)/second : 1

            dCa/dt = fp8_multiply_stochastic(Ca, alpha_ca)/second : 1

            Iconst : integer
            Vm_noise : volt
            Vreset : integer
            Vrest : integer
            alpha : integer (constant)
            alpha_refrac : integer (constant)
            alpha_syn : integer (constant)
            alpha_ca : integer (constant)
            '''
        self.threshold = 'Vm == Vthr'
        self.refractory = 'fp8_smaller_than(Vm, Vrest)==1'
        self.reset = '''
            Vm=Vreset;
            Vm_noise = 0*mV
            Ca = fp8_add_stochastic(Ca, Ca_inc)
            Ca = Ca + 128
            '''
        self.events = ParamDict({'active_Ca': 'Ca > 0'})
        self.namespace = ParamDict({
            'Vthr': 127,  # 480 in decimal
            'Ca_inc': 127
            })
        self.parameters = ParamDict({
            'Vreset':  '177',  # -0.5625 in decimal,
            'Vrest': '0',
            'Iconst': '0',
            'alpha': '55',  # 0.9375 in decimal,
            'alpha_refrac': '1',  # 0.00195312 in decimal
            'alpha_syn': '53',  # 0.8125 in decimal,
            'alpha_ca': '55',  # 0.9375 in decimal,
            'Vm': '0',
            'Ca': '0',
            'g': '0',
            'Vm_noise': '0*mV',
            })
