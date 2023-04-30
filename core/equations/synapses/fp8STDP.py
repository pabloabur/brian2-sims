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
                          'fp8_multiply(w_plast, w_factor)',
                          old_expr='fp8_multiply(weight, w_factor)')
        self.on_pre += ('w_change = fp8_multiply(eta, Ca_post)\n'
                        + 'w_change = fp8_multiply(184, w_change)\n'  # -1
                        + 'w_change = fp8_add(w_plast, w_change)\n'
                        + 'w_plast = int(w_change<=127)*w_change\n')

        self.on_post += ('w_change = fp8_multiply(eta, Ca_pre)\n'
                         + 'w_plast = fp8_add(w_plast, w_change)\n')

        self.parameters = ParamDict({**self.parameters,
                                     **{'w_plast': 82}}  # 10 in decimal
                                    )
        del self.parameters['weight']

        self.namespace = ParamDict({**self.namespace,
                                    **{'eta': 1}})  # Smallest learning rate
