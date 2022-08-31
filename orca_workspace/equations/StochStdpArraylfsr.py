from brian2.units import * 
StochStdpNew = {'model':
'''
        dI_syn/dt = int(I_syn*decay_syn/mA + decay_probability_syn)*mA/second : amp (clock-driven)
        decay_probability_syn = lfsr_timedarray( ((seed_syn+t) % lfsr_max_value_syn) + lfsr_init_syn ) / (2**lfsr_num_bits_syn): 1
        Iin{input_number}_post = I_syn * sign(weight)                           : amp (summed)

        decay_syn = tau_syn/(tau_syn + dt) : 1

        weight                : 1
        w_plast               : 1
        lfsr_max_value_syn : second
        seed_syn : second
        lfsr_init_syn : second
        gain_syn              : amp
        tau_syn               : second (constant)
        lfsr_num_bits_syn : 1 # Number of bits in the LFSR used
        
        dApre/dt = int(Apre * decay_stdp_Apre + decay_probability_Apre)/second : 1 (clock-driven)
        dApost/dt = int(Apost * decay_stdp_Apost + decay_probability_Apost)/second : 1 (clock-driven)
        decay_probability_Apre = lfsr_timedarray( ((seed_Apre+t) % lfsr_max_value_Apre) + lfsr_init_Apre ) / (2**lfsr_num_bits_Apre): 1
        decay_probability_Apost = lfsr_timedarray( ((seed_Apost+t) % lfsr_max_value_Apost) + lfsr_init_Apost ) / (2**lfsr_num_bits_Apost): 1

        decay_stdp_Apre = taupre/(taupre + dt) : 1
        decay_stdp_Apost = taupost/(taupost + dt) : 1

        seed_Apre : second
        lfsr_max_value_Apre : second
        lfsr_init_Apre : second
        lfsr_num_bits_Apre : 1
        seed_Apost : second
        lfsr_max_value_Apost : second
        lfsr_init_Apost : second
        lfsr_num_bits_Apost : 1
        w_max: 1 (constant)
        A_max: 1 (constant)
        dApre: 1 (constant)
        A_gain: 1 (constant)
        taupre : second (constant)
        taupost : second (constant)
        re_init_counter : 1

        counter_Apre : second
        counter_Apost : second
        Apre1_lfsr : 1
        Apre2_lfsr : 1
        Apost1_lfsr : 1
        Apost2_lfsr : 1
        cond_Apre1 : boolean
        cond_Apost1 : boolean
        cond_Apre2 : boolean
        cond_Apost2 : boolean
        stdp_thres : 1 (constant)
        lfsr_num_bits_condApre1 : 1
        lfsr_num_bits_condApre2 : 1
        lfsr_num_bits_condApost1 : 1
        lfsr_num_bits_condApost2 : 1
        seed_condApre1 : second
        seed_condApre2 : second
        seed_condApost1 : second
        seed_condApost2 : second
        lfsr_max_value_condApre1 : second
        lfsr_max_value_condApre2 : second
        lfsr_max_value_condApost1 : second
        lfsr_max_value_condApost2 : second
        lfsr_init_condApre1 : second
        lfsr_init_condApre2 : second
        lfsr_init_condApost1 : second
        lfsr_init_condApost2 : second
        ''',
'on_pre':
'''

        I_syn += gain_syn * weight * w_plast
        
        Apre += dApre
        counter_Apre += 1*ms
        Apre1_lfsr = lfsr_timedarray( ((seed_condApre1+counter_Apre) % lfsr_max_value_condApre1) + lfsr_init_condApre1 )
        cond_Apre1 = Apre1_lfsr < Apost
        Apre2_lfsr = lfsr_timedarray( ((seed_condApre2+counter_Apre) % lfsr_max_value_condApre2) + lfsr_init_condApre2 )
        cond_Apre2 = Apre2_lfsr <= stdp_thres
        Apre = clip(Apre, 0, A_max)
        w_plast = clip(w_plast - 1*int(lastspike_post!=lastspike_pre)*int(cond_Apre1)*int(cond_Apre2), 0, w_max)
        re_init_counter = re_init_counter + 1*int(lastspike_post!=lastspike_pre)*int(cond_Apre1)*int(cond_Apre2)
        ''',
'on_post':
'''

        
        Apost += dApre
        counter_Apost += 1*ms
        Apost1_lfsr = lfsr_timedarray( ((seed_condApost1+counter_Apost) % lfsr_max_value_condApost1) + lfsr_init_condApost1 )
        cond_Apost1 = Apost1_lfsr < Apre
        Apost2_lfsr = lfsr_timedarray( ((seed_condApost2+counter_Apost) % lfsr_max_value_condApost2) + lfsr_init_condApost2 )
        cond_Apost2 = Apost2_lfsr <= stdp_thres
        Apost = clip(Apost, 0, A_max)
        w_plast = clip(w_plast + 1*int(lastspike_post!=lastspike_pre)*int(cond_Apost1)*int(cond_Apost2), 0, w_max)
        re_init_counter = re_init_counter + 1*int(lastspike_post!=lastspike_pre)*int(cond_Apost1)*int(cond_Apost2)
        
''',
'parameters':
{
'weight' : '1',
'w_plast' : '1',
'gain_syn' : '1. * mamp',
'tau_syn' : '3. * msecond',
'lfsr_num_bits_syn' : '6',
'taupre' : '3. * msecond',
'taupost' : '3. * msecond',
'w_max' : '15',
'A_max' : '15',
'A_gain' : '4',
'dApre' : '15',
'lfsr_num_bits_Apre' : '6',
'lfsr_num_bits_Apost' : '6',
'counter_Apre': '0 * msecond',
'counter_Apost': '0 * msecond',
'stdp_thres': '15',
'lfsr_num_bits_condApre1': '4',
'lfsr_num_bits_condApre2': '4',
'lfsr_num_bits_condApost1': '4',
'lfsr_num_bits_condApost2': '4',
're_init_counter': '0',
}
}
