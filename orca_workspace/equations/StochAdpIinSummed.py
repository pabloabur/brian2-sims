from brian2.units import * 
StochAdpIinSummed = {'model':
'''
        dI_syn/dt = int(I_syn*decay_syn/mA + decay_probability_syn)*mA/second : amp (clock-driven)
        decay_probability_syn = rand() : 1 (constant over dt)
        Iin{input_number}_post = I_syn * sign(weight)                           : amp (summed)

        decay_syn = tausyn/(tausyn + dt) : 1

        weight                : 1
        w_plast               : 1
        lfsr_max_value_syn : second
        seed_syn : second
        lfsr_init_syn : second
        gain_syn              : amp
        tausyn               : second (constant)
        inh_learning_rate: 1 (constant, shared)
        variance_th: 1 (constant)
        delta_w : 1
        
         ''',
'on_pre':
'''

        I_syn += gain_syn * weight * w_plast
        
        delta_w = inh_learning_rate * (0.5 - normalized_activity_proxy_post)
        w_plast = clip(w_plast + delta_w, 0, 15)
         ''',
'on_post':
'''

        
         
''',
'parameters':
{
'weight' : '1',
'w_plast' : '1',
'gain_syn' : '1. * mamp',
'tausyn' : '3. * msecond',
'inh_learning_rate' : '0.1',
'variance_th' : '0.67',
}
}
