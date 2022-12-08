""" Model deterministic LIF with 8-bit fixed-point precision"""
from brian2.units import * 
from core.equations.base_equation import BaseNeuron, ParamDict
from core.equations.neurons.int4LIF import int4LIF

class int8LIF(int4LIF):
    def __init__(self):
        super().__init__()
        self.modify_model('model',
                          'deterministic_decay',
                          old_expr='stochastic_decay')
        self.modify_model('namespace', 255, key='Vm_max')
        self.modify_model('namespace', 127, key='g_max')
        self.modify_model('namespace', -128, key='g_min')
        self.modify_model('namespace', 51, key='Vrest')
        self.modify_model('parameters', 255, key='Vthr')
        self.modify_model('parameters', 51, key='Vm')
