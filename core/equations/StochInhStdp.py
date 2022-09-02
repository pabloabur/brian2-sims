from brian2.units import * 
StochInhStdp = {'model':
'''
        weight                : 1
        w_plast               : 1
        w_max                 : 1
        gain_syn              : amp
        variance_th: 1 (constant)
        delta_w : 1

        dApre/dt = int(Apre * decay_stdp_Apre + decay_probability_Apre)/second : 1 (clock-driven)
        dApost/dt = int(Apost * decay_stdp_Apost + decay_probability_Apost)/second : 1 (clock-driven)
        decay_probability_Apre = rand() : 1 (constant over dt)
        decay_probability_Apost = rand() : 1 (constant over dt)

        decay_stdp_Apre = taupre/(taupre + dt) : 1
        decay_stdp_Apost = taupost/(taupost + dt) : 1

        A_max: 1 (constant)
        A_gain: 1 (constant)
        taupre : second (constant)
        taupost : second (constant)
        
        rand_int_Apre1 : 1
        rand_int_Apre2 : 1
        rand_int_Apost1 : 1
        rand_int_Apost2 : 1
        rand_num_bits_syn : 1 # Number of bits of random number generated for As
        stdp_thres : 1 (constant)
         ''',
'on_pre':
'''

        I_syn_post += (gain_syn * weight * w_plast)
        
        Apre += 15
        Apre = clip(Apre, 0, A_max)
        rand_int_Apre1 = ceil(rand() * (2**rand_num_bits_syn-1))
        rand_int_Apre2 = ceil(rand() * (2**rand_num_bits_syn-1))
        delta_w  = 1 * sign(Apost - variance_th) * int(lastspike_post!=lastspike_pre)*int(rand_int_Apre1 < abs(Apost - variance_th))*int(rand_int_Apre2 <= stdp_thres)
        w_plast = clip(w_plast + delta_w, 0, w_max)
         ''',
'on_post':
'''
        Apost += 15
        Apost = clip(Apost, 0, A_max)
        rand_int_Apost1 = ceil(rand() * (2**rand_num_bits_syn-1))
        rand_int_Apost2 = ceil(rand() * (2**rand_num_bits_syn-1))
        delta_w  = 1 * int(lastspike_post!=lastspike_pre)*int(rand_int_Apost1 < Apre)*int(rand_int_Apost2 <= stdp_thres)
        w_plast = clip(w_plast + delta_w, 0, w_max)

        
         
''',
'parameters':
{
'weight' : '1',
'w_plast' : '1',
'gain_syn' : '1. * mamp',
'variance_th' : '2',
'taupre': '20 * msecond',
'taupost': '20 * msecond',
'stdp_thres': '1',
'rand_num_bits_syn': '4'
}
}
