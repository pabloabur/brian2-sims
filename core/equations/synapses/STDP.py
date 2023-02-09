""" Conventional synapse model
    """

from brian2 import *

from core.equations.synapses.CUBA import CUBA, ParamDict

class STDP(CUBA):
    def __init__(self):
        super().__init__()
        self.model += ('di_trace/dt = alpha_itrace*i_trace/second : 1 (clock-driven)\n'
                       + 'dj_trace/dt = alpha_jtrace*j_trace/second : 1 (clock-driven)\n'
                       + 'alpha_itrace : 1 (constant)\n'
                       + 'alpha_jtrace : 1 (constant)\n'
                       + 'tau_itrace : second (constant)\n'
                       + 'tau_jtrace : second (constant)\n'
                       + 'w_plast : volt\n')

        self.modify_model('on_pre', 'g += w_plast', 'g += weight')
        self.modify_model('model', '', 'weight : volt')
        self.on_pre += ('i_trace += 1\n'
                         + 'w_plast = clip(w_plast - eta*j_trace, 0*volt, w_max)\n')
        self.on_post += ('j_trace += 1\n'
                         + 'w_plast = clip(w_plast + eta*i_trace, 0*volt, w_max)\n')
        self.namespace = ParamDict({**self.namespace,
                                    **{'w_max': 100*mV,
                                       'eta': 1*mV,
                                       }})
        del self.parameters['weight']
        self.parameters = ParamDict({**self.parameters,
                                     **{'w_plast': 0.5*mV,
                                        'tau_itrace': 20*ms,
                                        'tau_jtrace': 20*ms,
                                        'alpha_itrace': 'tau_itrace/(dt + tau_itrace)',
                                        'alpha_jtrace': 'tau_jtrace/(dt + tau_jtrace)'}})
