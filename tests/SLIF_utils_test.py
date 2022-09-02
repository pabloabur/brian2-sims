from brian2 import *
from SLIF_utils import neuron_rate

def test_rates():
    pg = PoissonGroup(2, 20*Hz)
    mon = SpikeMonitor(pg)

    run(2*second)

    nr = neuron_rate(mon, kernel_len=10*ms, kernel_var=10*ms,
                     simulation_dt=defaultclock.dt,
                     #interval=[0*ms, 2000*ms],
                     smooth=True,
                     trials=1)

    figure()
    plot(mon.t, mon.i, '.', label='spikes')
    xlabel('second')
    ylabel('Hz')

    step(nr['t'], nr['rate'][0], where='pre', label='rate 1')
    plot(nr['t'], nr['smoothed'][0], label='smoothed rate 1')
    #step(nr['t'], nr['rate'][1], where='pre', label='rate 2')
    #plot(nr['t'], nr['smoothed'][1], label='smoothed rate 2')
    legend()
    show()

test_rates()
