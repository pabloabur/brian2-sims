""" Conventional synapse model
    """

from brian2 import *
from core.equations.base_equation import BaseSynapse, ParamDict

class CUBA(BaseSynapse):
    def __init__(self):
        super().__init__()
        self.model = '''
            dg/dt = alpha_syn*g/second : volt (clock-driven)
            gtot0_post = g*w_factor : volt (summed)

            alpha_syn : 1 (constant)
            weight : volt
            tau_syn : second
            '''
        self.on_pre = '''
            g += weight
            '''
        self.namespace = ParamDict({
            'w_factor': 1,
            })
        self.parameters =  ParamDict({
            'tau_syn': '5*ms',
            'alpha_syn': 'tau_syn/(dt + tau_syn)',
            'delay': '0*ms',
            'weight': '1*mV'
            })
