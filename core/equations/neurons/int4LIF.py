""" Model stochastic LIF with 4-bit fixed-point precision"""
from brian2.units import * 
from core.equations.base_equation import BaseNeuron, ParamDict

class int4LIF(BaseNeuron):
    def __init__(self):
        super().__init__()
        self.model = '''
            dVm/dt = summed_decay / second : 1
            # In hardware a conversion to signed current to unsigned is necessary
            summed_decay = clip(decay_term + gtot*int(not_refractory), 0, Vm_max) : integer
            decay_term = normal_decay*int(not_refractory) + refrac_decay*int(not not_refractory) : integer

            normal_decay = clip(stochastic_decay(Vm, vm_decay_numerator), Vrest, Vm_max) : 1 (constant over dt)
            # Decays distance to Vrest to then update it
            refrac_decay = Vrest - stochastic_decay(Vrest - Vm, refrac_decay_numerator) : 1 (constant over dt)

            gtot = clip(g + Iconst, g_min, g_max) : integer
            dg/dt = aux_g/second : 1
            aux_g = stochastic_decay(g_clipped, syn_decay_numerator) : 1 (constant over dt)
            g_clipped = clip(g, g_min, g_max) : integer

            Iconst : integer
            Vm_noise : integer
            Vthr : integer (constant)
            vm_decay_numerator : 1 (constant) # L in L/256 describing decay
            refrac_decay_numerator : 1 (constant) # L in L/256 describing decay
            syn_decay_numerator : 1 (constant) # L in L/256 describing decay
            '''
        self.threshold = 'Vm == Vthr'
        self.refractory = 'Vm<Vrest'
        self.reset = '''
            Vm=Vreset;
            Vm_noise = 0
            '''
        self.namespace = ParamDict({
            'Vm_max': 15,
            'g_max': 7,
            'g_min': -8,
            'Vreset': 0,
            'Vrest': 3,
            })
        self.parameters = ParamDict({
            'Iconst': '0',
            'Vthr': 15,
            'vm_decay_numerator': 240,  # 240/256 ~ 0.9375
            'refrac_decay_numerator': 4,  # to approximate 2ms refractory period
            'syn_decay_numerator': 213,  # ~ 0.832
            'Vm': '3',
            'g': '0',
            'Vm_noise': '0',
            })
