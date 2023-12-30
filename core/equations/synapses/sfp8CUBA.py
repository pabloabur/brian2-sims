""" 8-bit floating point implementation of CUBA """
from brian2.units import *
from core.equations.base_equation import BaseSynapse, ParamDict

class sfp8CUBA(BaseSynapse):
    def __init__(self):
        super().__init__()
        self.model = '''
            weight : integer
            '''
        self.on_pre = ParamDict({'pre':
            'g_post = fp8_add_stochastic(g_post, fp8_multiply_stochastic(weight, w_factor))'})
        self.namespace = ParamDict({
            'w_factor': 56,  # 1 in decimal
            })
        self.parameters = ParamDict({
            'delay': '0*ms',
            'weight': '82'  # 10 in decimal
            })
