from brian2.units import *

song_syn = {
    'model':'''
        w_plast : 1
        dApre/dt = -Apre / taupre : 1 (event-driven)
        dApost/dt = -Apost / taupost : 1 (event-driven)

        taupre : second (constant)
        taupost : second (constant)
        w_max : 1
        deltaApre : 1 (constant)
        deltaApost : 1 (constant)
        ''',
    'on_pre':'''
        ge_post += w_plast
        Apre += deltaApre
        w_plast = clip(w_plast - Apost, 0, w_max)
        ''',
    'on_post':'''
        Apost += deltaApost
        w_plast = clip(w_plast + Apre, 0, w_max)
        ''',
    'parameters':{
        'taupre': '20*msecond',
        'taupost': '20*msecond',
        'w_max': '.01',
        'deltaApre': '.0001',
        'deltaApost': '.000105'
        }
    }
