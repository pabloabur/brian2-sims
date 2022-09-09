""" synapse model with STDP and heterosynaptic mechanism
    """

from brian2 import *

from core.equations.synapses.STDP import STDP, ParamDict

class hSTDP(STDP):
    def __init__(self):
        super().__init__()
        self.model += (
            'incoming_weights_post = w_plast : volt (summed)\n'
            + 'outgoing_weights_pre = w_plast : volt (summed)\n'
            + 'incoming_factor = incoming_weights_post - w_lim : volt\n'
            + 'outgoing_factor = outgoing_weights_pre - w_lim : volt\n'
            + 'heterosyn_factor = int(incoming_factor > 0*volt)*incoming_factor + int(outgoing_factor > 0*volt)*outgoing_factor : volt\n')

        self.namespace = ParamDict({**self.namespace, **{'w_lim': 1*mV,
                                                         'h_eta': .01}})
