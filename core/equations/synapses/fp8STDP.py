from brian2.units import *
from core.equations.synapses.fp8CUBA import fp8CUBA, ParamDict


class fp8STDP(fp8CUBA):
    def __init__(self):
        """ Implementation of STDP with minifloat.

        Notes
        -----
        Plastic weight is a positive value. Sums higher than the maximum value
        overflow so there is no need to adjust it. However, subtracting the
        minimum results in values higer than 127 (in minifloat), in which case
        it needs to be reset to the minimum.
        """
        super().__init__()
        self.model += 'w_plast : integer\n'
        self.modify_model('model', '', old_expr='weight : integer')

        self.modify_model('on_pre',
                          'g_post = fp8_add(g_post, fp8_multiply(w_plast, w_factor))',
                          key='pre')
        self.on_pre = ParamDict({
                        **self.on_pre,
                        **{'stdp_fanout': '''
                            delta_w = int(Ca_pre<128 and Ca_post>128)*fp8_multiply(eta, Ca_pre)+ int(Ca_pre>128 and Ca_post<128)*(128 + fp8_multiply(eta, Ca_post))
                            w_plast = fp8_add(w_plast, delta_w)'''}})
        self.on_event = ParamDict({'pre': 'spike', 'stdp_fanout': 'active_Ca'})

        self.parameters = ParamDict({**self.parameters,
                                     **{'w_plast': 82}}  # 10 in decimal
                                    )
        del self.parameters['weight']

        self.namespace = ParamDict({**self.namespace,
                                    **{'eta': 1}})  # Smallest learning rate
