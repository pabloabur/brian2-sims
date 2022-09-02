from brian2.units import * 
StochAdpIin = {'model':
'''
        weight                : 1
        w_plast               : 1
        w_max                 : 1 (constant)
        lfsr_max_value_syn : second
        seed_syn : second
        lfsr_init_syn : second
        gain_syn              : amp
        inh_learning_rate: 1 (constant, shared)
        variance_th: 1 (constant)
        delta_w : 1
        
         ''',
'on_pre':
'''

        I_syn_post += (gain_syn * weight * w_plast)
        
        delta_w = inh_learning_rate * (0.5 - normalized_activity_proxy_post)
        w_plast = clip(w_plast + delta_w, 0, w_max)
         ''',
'on_post':
'''

        
         
''',
'parameters':
{
'weight' : '1',
'w_plast' : '1',
'gain_syn' : '1. * mamp',
'inh_learning_rate' : '0.1',
'variance_th' : '0.67',
}
}
