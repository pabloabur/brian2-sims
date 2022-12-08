""" Model deterministic CUBA with 8-bit fixed-point precision"""
from brian2.units import * 
from core.equations.base_equation import BaseSynapse, ParamDict
from core.equations.synapses.int4CUBA import int4CUBA

class int8CUBA(int4CUBA):
    def __init__(self):
        super().__init__()
        self.modify_model('parameters', 64, key='weight')
