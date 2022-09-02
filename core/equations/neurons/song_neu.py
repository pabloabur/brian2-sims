from brian2.units import *

song_neu = {
    'model':'''
        dVm/dt = (ge * (Ee-Vm) + El - Vm) / taum : volt
        dge/dt = -ge / taue : 1

        refP : second (constant)
        taue : second (constant)
        taum : second (constant)
        Ee : volt (constant)
        El : volt (constant)
        vt : volt (constant)
        vr : volt (constant)
        ''',
    'threshold':'''Vm>vt''',
    'reset':'''
        Vm = vr
        ''',
    'parameters':{
        'refP': '0.*second',
        'Ee': '0*mvolt',
        'vt': '-54*mvolt',
        'vr': '-60*mvolt',
        'El': '-74*mvolt',
        'taum': '10*msecond',
        'taue': '5*msecond'
        }
    }
