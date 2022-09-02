from brian2.units import * 
StochInhStdpSummed = {'model':
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
        variance_th: 1 (constant)
        delta_w : 1

        dApre/dt = int(Apre * decay_stdp_Apre + decay_probability_Apre)/second : 1 (clock-driven)
        dApost/dt = int(Apost * decay_stdp_Apost + decay_probability_Apost)/second : 1 (clock-driven)
        decay_probability_Apre = rand() : 1 (constant over dt)
        decay_probability_Apost = rand() : 1 (constant over dt)

        decay_stdp_Apre = taupre/(taupre + dt) : 1
        decay_stdp_Apost = taupost/(taupost + dt) : 1

        A_max: 1 (constant)
        dApre: 1 (constant)
        A_gain: 1 (constant)
        taupre : second (constant)
        taupost : second (constant)
        
        rand_int_Apre1 : 1
        rand_int_Apre2 : 1
        rand_int_Apost1 : 1
        rand_int_Apost2 : 1
        rand_num_bits_Apre : 1 # Number of bits of random number generated for Apre
        rand_num_bits_Apost : 1 # Number of bits of random number generated for Apost
        stdp_thres : 1 (constant)
         ''',
'on_pre':
'''

        I_syn += gain_syn * weight * w_plast
        
        Apre += 15
        Apre = clip(Apre, 0, 15)
        rand_int_Apre1 = ceil(rand() * (2**rand_num_bits_Apre-1))
        rand_int_Apre2 = ceil(rand() * (2**rand_num_bits_Apre-1))
        delta_w  = 1 * sign(Apost - variance_th) * int(lastspike_post!=lastspike_pre)*int(rand_int_Apre1 < abs(Apost - variance_th))*int(rand_int_Apre2 <= stdp_thres)
        w_plast = clip(w_plast + delta_w, 0, 31)
         ''',
'on_post':
'''
        Apost += 15
        Apost = clip(Apost, 0, 15)
        rand_int_Apost1 = ceil(rand() * (2**rand_num_bits_Apost-1))
        rand_int_Apost2 = ceil(rand() * (2**rand_num_bits_Apost-1))
        delta_w  = 1 * int(lastspike_post!=lastspike_pre)*int(rand_int_Apost1 < Apre)*int(rand_int_Apost2 <= stdp_thres)
        w_plast = clip(w_plast + delta_w, 0, 31)

        
         
''',
'parameters':
{
'weight' : '1',
'w_plast' : '1',
'gain_syn' : '1. * mamp',
'tausyn' : '3. * msecond',
'variance_th' : '2',
'taupre': '20 * msecond',
'taupost': '20 * msecond',
'stdp_thres': '1',
'rand_num_bits_Apre': '4',
'rand_num_bits_Apost': '4'
}
}
