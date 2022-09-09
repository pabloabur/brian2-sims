""" synapse model with STDP and heterosynaptic mechanism
    """

from brian2 import *

from core.equations.synapses.STDP import STDP, ParamDict

class iSTDP(STDP):
    def __init__(self):
        super().__init__()

        self.modify_model('on_pre',
                          'w_plast = clip(w_plast + (j_trace-target_rate)*eta, 0*volt, w_max)',
                          'w_plast = clip(w_plast - eta*j_trace, 0*volt, w_max)')

        self.namespace = ParamDict({**self.namespace, **{'target_rate': .12}})
        self.namespace['w_factor'] = -1
