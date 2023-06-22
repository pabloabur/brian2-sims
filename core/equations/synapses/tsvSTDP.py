from brian2.units import *
from core.equations.synapses.tsvCUBA import tsvCUBA, ParamDict


class tsvSTDP(tsvCUBA):
    def __init__(self):
        """ Implementation of STDP with three state variable scheme.

        """
        super().__init__()
        self.modify_model('model', 'w_plast ', old_expr='weight')
        self.modify_model('on_pre', 'w_plast ', old_expr='weight')

        self.parameters = ParamDict({**self.parameters,
                                     **{'w_plast': 0.5*mV}}
                                    )
        del self.parameters['weight']

        self.namespace = ParamDict({**self.namespace,
                                    **{'eta': 1*mV}})
