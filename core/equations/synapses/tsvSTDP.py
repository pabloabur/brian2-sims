from brian2.units import *
from core.equations.synapses.tsvCUBA import tsvCUBA, ParamDict


class tsvSTDP(tsvCUBA):
    def __init__(self):
        """ Implementation of STDP with three state variable scheme.

        """
        super().__init__()
        self.modify_model('model', 'w_plast ', old_expr='weight')
        self.modify_model('on_pre', 'g_post += (w_plast*w_factor)', key='pre')
        self.on_pre = ParamDict({
            **self.on_pre,
            **{'stdp_fanout': '''
                delta_w = int(Ca_pre>0 and Ca_post<0)*(eta*Ca_pre) - int(Ca_pre<0 and Ca_post>0)*(eta*Ca_post)
                w_plast = clip(w_plast + delta_w, 0*volt, w_max)'''}})
        self.on_event = ParamDict({'pre': 'spike', 'stdp_fanout': 'active_Ca'})

        self.parameters = ParamDict({**self.parameters,
                                     **{'w_plast': 0.5*mV}}
                                    )
        del self.parameters['weight']

        self.namespace = ParamDict({**self.namespace,
                                    **{'w_max': 100*mV,
                                       'eta': 1*mV}})
