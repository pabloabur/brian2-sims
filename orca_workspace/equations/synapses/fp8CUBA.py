""" 8-bit floating point implementation of CUBA """
from brian2.units import *
from equations.base_equation import BaseSynapse, ParamDict

class fp8CUBA(BaseSynapse):
    def __init__(self):
        super().__init__()
        self.model = '''
            weight : integer
            '''
        self.on_pre = '''
            g_post = fp8_add(g_post, fp8_multiply(weight, w_factor))
            '''
        self.namespace = ParamDict({
            'w_factor': 56,  # 1 in decimal
            })
        self.parameters = ParamDict({
            'delay': '0*ms',
            'weight': 'decimal2minifloat(10)'
            })
