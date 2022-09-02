from brian2.units import * 
StochDecayMath.py = {'model':
'''
        dVm/dt = (int(not refrac)*int(normal_decay) + int(refrac)*int(refractory_decay))*mV/second : volt
        normal_decay = clip((decay_rate*Vm + (1-decay_rate)*(Vrest + g_psc*I) + Vm_noise)/mV + decay_probability, Vrest/mV, Vm_max) : 1
        refractory_decay = (decay_rate_refrac*Vm + (1-decay_rate_refrac)*Vrest)/mV + decay_probability : 1
        decay_probability = rand() : 1 (constant over dt)

        dI_syn/dt = int(clip(I_syn + Iconst, I_min, I_max)*decay_syn/mA + decay_probability_syn)*mA/second : amp
        decay_probability_syn = rand() : 1 (constant over dt)
        decay_syn = tausyn/(tausyn + dt) : 1

        I = clip(I_syn, I_min, I_max) : amp
        decay_rate = tau/(tau + dt)                      : 1
        decay_rate_refrac = refrac_tau/(refrac_tau + dt) : 1
        refrac = Vm<Vrest                                    : boolean

        g_psc                : ohm    (constant) # Gain of post synaptic current
        Iconst  : amp                         # constant input current
        Vm_noise          : volt
        tau               : second (constant)
        tausyn               : second (constant)
        refrac_tau        : second (constant)
        refP              : second
        Vthr              : volt   (constant)
        Vm_max            : 1      (constant)
        I_min            : amp      (constant)
        I_max            : amp      (constant)
        Vrest             : volt   (constant)
        Vreset            : volt   (constant)


    
        x : 1 (constant) # x location on 2d grid
        y : 1 (constant) # y location on 2d grid
        
         Iin = Iin0  : amp # input currents

         Iin0 : amp
''',
'threshold':
'''Vm>=Vthr''',
'reset':
'''
        Vm=Vreset;
        Vm_noise = 0*mV ''',
'parameters':
{
'Vthr' : '15. * mvolt',
'Vm_max' : '15',
'I_min' : '-15. * mamp',
'I_max' : '15. * mamp',
'Vrest' : '3. * mvolt',
'Vreset' : '0. * volt',
'Iconst' : '0. * amp',
'g_psc' : '1. * ohm',
'tau' : '19. * msecond',
'refrac_tau' : '2. * msecond',
'refP' : '0. * second',
'Vm_noise' : '0. * volt',
'tausyn' : '3. * msecond',
}
}
