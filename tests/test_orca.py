"""
Created on Thu Feb 3 7:28:46 2022

@author: pablo
"""
import unittest
import numpy as np

from brian2 import SpikeGeneratorGroup, run, ms, defaultclock, prefs, \
    StateMonitor, set_device, device

from brian2tools import brian_plot

from core.equations.neurons.fp8LIF import fp8LIF
from core.equations.synapses.fp8CUBA import fp8CUBA
from core.builder.groups_builder import create_synapses, create_neurons
from core.utils.misc import DEFAULT_FUNCTIONS
from core.equations.base_equation import ParamDict

#prefs.codegen.target = "numpy"
set_device('cpp_standalone')

class TestOrca(unittest.TestCase):

    def test_addition(self):
        device.reinit()
        device.activate()
        defaultclock.dt = 1*ms

        # Each synapse represents one test: g <- weight + g
        ws = [135, 3, 131, 10,  5, 143,  9, 138, 12, 28,  20, 232, 238, 68, 145, 119, 120]
        g0 = [  8, 1,   4,  1, 11,  16,  1,  10,  8, 12, 139, 239, 239, 44,  56, 113, 119]
        gn = [  1, 4,   1, 11, 16,   1, 10,   0, 18, 31,  13, 244, 246, 70,  55, 124, 127]
        n_input = len(ws)

        neu = fp8LIF()
        neu.modify_model('parameters', '56', key='alpha_syn')
        neu.modify_model('parameters', g0, key='g')
        neu = create_neurons(n_input, neu, raise_warning=True)

        indices = range(n_input)
        times = [1*ms] * n_input
        inp = SpikeGeneratorGroup(n_input, indices, times)

        syn = fp8CUBA()
        syn.modify_model('parameters', ws, key='weight')
        syn.modify_model('connection', 'i', key='j')
        syn = create_synapses(inp, neu, syn, raise_warning=True)

        run(3*ms)
        res = neu.g[:]
        for i in range(len(res)):
            self.assertEqual(res[i], gn[i], f'{ws[i]}+{g0[i]} should be '
                                            f'{gn[i]}, but was {res[i]}')

    def test_multiplication(self):
        device.reinit()
        device.activate()
        defaultclock.dt = 1*ms

        # Each synapse represents one test: g <- weight + g
        ws = [62, 53, 48, 176, 176, 63, 16, 135, 63, 63, 119, 95]
        w0 = [74, 18, 52,  52, 180, 63,  1,   7,  7,  1, 113, 81]
        gn = [81, 16, 44, 172,  44, 70,  0,   0, 13,  2, 127, 120]
        n_input = len(ws)

        neu = fp8LIF()
        neu.modify_model('parameters', '56', key='alpha_syn')
        neu = create_neurons(n_input, neu, raise_warning=True)

        indices = range(n_input)
        times = [1*ms] * n_input
        inp = SpikeGeneratorGroup(n_input, indices, times)

        syn = fp8CUBA()
        # Makes w_factor not global to make simulations easier
        del syn.namespace['w_factor']
        syn.model += 'w_factor : integer\n'
        syn.parameters = ParamDict({**syn.parameters, **{'w_factor': w0}})
        syn.modify_model('parameters', ws, key='weight')
        syn.modify_model('connection', 'i', key='j')
        syn = create_synapses(inp, neu, syn, raise_warning=True)

        run(3*ms)
        res = neu.g[:]
        for i in range(len(res)):
            self.assertEqual(res[i], gn[i], f'{ws[i]}*{w0[i]} should be '
                                            f'{gn[i]}, but was {res[i]}')

if __name__ == '__main__':
    unittest.main(verbosity=2, exit=False)
