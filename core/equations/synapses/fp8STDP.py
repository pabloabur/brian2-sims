from brian2.units import *
from core.equations.synapses.fp8CUBA import fp8CUBA, ParamDict

class fp8STDP(fp8CUBA):
    def __init__(self):
        super().__init__()
        self.model += 'w_plast : integer\n'
        self.modify_model('model', '', old_expr='weight : integer')

        self.modify_model('on_pre',
                          'fp8_multiply(w_plast, w_factor)',
                          old_expr='fp8_multiply(weight, w_factor)')
        self.on_pre += ('weighted_Ca = fp8_multiply(eta, Ca_post)\n'
                        + 'weighted_Ca = fp8_multiply(184, weighted_Ca)\n' # -1
                        + 'w_plast = clip(fp8_add(w_plast, weighted_Ca), 0, inf)\n')

        self.on_post += ('weighted_Ca = fp8_multiply(eta, Ca_pre)\n'
                         + 'w_plast = clip(fp8_add(w_plast, weighted_Ca), 0, inf)\n')

        self.parameters = ParamDict({**self.parameters,
                                     **{'w_plast': 82}} # 10 in decimal
                                     )
        del self.parameters['weight']

        self.namespace = ParamDict({**self.namespace,
                                    **{'eta': 1}}) # Smallest learning rate
