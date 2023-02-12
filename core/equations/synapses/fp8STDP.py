from brian2.units import *
from core.equations.synapses.fp8CUBA import fp8CUBA, ParamDict

class fp8STDP(fp8CUBA):
    def __init__(self):
        super().__init__()
        self.model += 'w_plast : volt\n'
        self.modify_model('model', '', old_expr='weight : volt')

        self.modify_model('on_pre',
                          'fp8_multiply(w_plast, w_factor)',
                          old_expr='fp8_multiply(weight, w_factor)')
        self.on_pre += 'w_plast = clip(w_plast - eta*Ca_post, 0, inf)\n'

        self.on_post += 'w_plast = clip(w_plast + eta*Ca_pre, 0, inf)\n'

        self.parameters = ParamDict({**self.parameters,
                                     **{'w_plast': 82}} # 10 in decimal
                                     )
        del self.parameters['weight']
