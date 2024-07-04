from brian2.units import *
from core.equations.synapses.sfp8STDP import sfp8STDP, ParamDict


class sfp8iSTDP(sfp8STDP):
    def __init__(self):
        """ Implementation of inhibitory STDP with minifloat.
        """
        super().__init__()
        self.modify_model('namespace', 184, key='w_factor')
        self.model += 'target_rate : integer\n'

        self.modify_model('on_pre',
                          '''delta_w = int(Ca_pre<128 and Ca_post>128)*Ca_pre + int(Ca_pre>128 and Ca_post<128)*fp8_add_stochastic(Ca_post, target_rate)
                             delta_w = fp8_multiply_stochastic(delta_w, eta)
                             w_plast = fp8_add_stochastic(w_plast, delta_w)
                             w_plast = int(w_plast<128)*w_plast
                             ''',
                          key='stdp_fanout')

        self.parameters = ParamDict({**self.parameters,
                                     **{'target_rate': 236}}  # -20 in decimal
                                    )
