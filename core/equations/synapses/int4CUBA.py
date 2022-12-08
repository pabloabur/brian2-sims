""" Model stochastic CUBA with 4-bit fixed-point precision"""
from brian2.units import * 
from core.equations.base_equation import BaseSynapse, ParamDict

class int4CUBA(BaseSynapse):
    def __init__(self):
        super().__init__()
        self.model = '''
            weight : integer
            '''
        self.on_pre = '''
            g_post += weight * w_factor
            '''
        self.namespace = ParamDict({
            'w_factor': 1,
            })
        self.parameters = ParamDict({
            'delay': '0*ms',
            'weight': '4',
            })
