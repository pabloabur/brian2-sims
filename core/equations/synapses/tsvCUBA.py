""" Conventional synapse model
    """

from brian2 import *
from core.equations.base_equation import BaseSynapse, ParamDict

class tsvCUBA(BaseSynapse):
    def __init__(self):
        super().__init__()
        self.model = '''
            weight : volt
            '''
        self.on_pre = ParamDict({'pre': 'g_post += (weight*w_factor)'})
        self.namespace = ParamDict({
            'w_factor': 1,
            })
        self.parameters =  ParamDict({
            'delay': '0*ms',
            'weight': '1*mV'
            })
