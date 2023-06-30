"""
Created on Thu Feb 3 7:28:46 2022

@author: pablo
"""
import unittest
import numpy as np

from brian2 import SpikeGeneratorGroup, run, ms, defaultclock, prefs, \
    DEFAULT_FUNCTIONS, seed

from core.equations.neurons.fp8LIF import fp8LIF
from core.equations.synapses.fp8CUBA import fp8CUBA
from core.builder.groups_builder import create_synapses, create_neurons
from core.equations.base_equation import ParamDict
from core.utils.misc import minifloat2decimal, decimal2minifloat

from core.utils.misc import stochastic_decay, fp8_multiply, fp8_add,\
    fp8_smaller_than, deterministic_decay, fp8_add_stochastic
DEFAULT_FUNCTIONS.update({'stochastic_decay': stochastic_decay,
                          'fp8_multiply': fp8_multiply,
                          'fp8_add': fp8_add,
                          'fp8_add_stochastic': fp8_add_stochastic,
                          'fp8_smaller_than': fp8_smaller_than,
                          'deterministic_decay': deterministic_decay})


class TestOrca(unittest.TestCase):

    def test_addition(self):
        prefs.codegen.target = "numpy"
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

    def test_addition_stochastic(self):
        """ Seed did not work with analogous cpp function, so I tested with
            averages. It is not ideal, but I did not want to spend time dealing
            with brian2's cpp_standalone.
        """
        prefs.codegen.target = "numpy"
        defaultclock.dt = 1*ms

        # Each synapse represents one test: g <- weight + g
        n_trials = 10000
        ws = ([135 for _ in range(n_trials)]
              + [232 for _ in range(n_trials)]
              + [238 for _ in range(n_trials)]
              + [68 for _ in range(n_trials)]
              + [145 for _ in range(n_trials)]
              + [39 for _ in range(n_trials)]
              + [31 for _ in range(n_trials)]
              + [22 for _ in range(n_trials)]
              + [14 for _ in range(n_trials)]
              + [32 for _ in range(n_trials)]
              + [56 for _ in range(n_trials)]
              + [56 for _ in range(n_trials)]
              + [48 for _ in range(n_trials)])
        g0 = ([8 for _ in range(n_trials)]
              + [239 for _ in range(n_trials)]
              + [239 for _ in range(n_trials)]
              + [44 for _ in range(n_trials)]
              + [56 for _ in range(n_trials)]
              + [15 for _ in range(n_trials)]
              + [15 for _ in range(n_trials)]
              + [15 for _ in range(n_trials)]
              + [23 for _ in range(n_trials)]
              + [33 for _ in range(n_trials)]
              + [88 for _ in range(n_trials)]
              + [112 for _ in range(n_trials)]
              + [120 for _ in range(n_trials)])
        ref = [0, .5, .5, .5, .25, .44, .37, .75, 0, .5, .5, .0625, .015625]
        tol = [.02 for _ in range(len(ref) - 2)] + [.005, .0025]
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
        syn.modify_model('on_pre', 'fp8_add_stochastic', old_expr='fp8_add')
        syn = create_synapses(inp, neu, syn, raise_warning=True)

        run(3*ms)
        res = neu.g[:]
        avg_res = [np.average(res[i*n_trials:(i+1)*n_trials])
                   for i in range(len(ws))]
        avg_dec = [x % 1 for x in avg_res]
        for i in range(len(ref)):
            self.assertAlmostEqual(avg_dec[i], ref[i], delta=tol[i])

    def test_multiplication(self):
        prefs.codegen.target = "numpy"
        defaultclock.dt = 1*ms

        # Each synapse represents one test: g <- weight + g
        ws = [62, 53, 48, 176, 176, 63, 16, 135, 63, 63, 119, 95, 7, 7, 7]
        w0 = [74, 18, 52,  52, 180, 63,  1,   7,  7,  1, 113, 81, 55, 124, 64]
        gn = [81, 16, 44, 172,  44, 70,  0,   0, 13,  2, 127, 120, 7, 74, 14]
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

    def test_conversions(self):
        self.assertEqual(minifloat2decimal(55), 0.9375, 'failed to convert int')

        self.assertEqual(minifloat2decimal(127.0), 480, 'failed to convert float')

        list_res = minifloat2decimal([56, 97.0])
        list_ref = [1, 36]
        for res, ref in zip(list_res, list_ref):
            self.assertEqual(res, ref, 'failed to convert list')

        list_res = minifloat2decimal(np.array([58.1, 66]))
        list_ref = [1.25, 2.5]
        for res, ref in zip(list_res, list_ref):
            self.assertEqual(res, ref, 'failed to convert np.ndarray')

        self.assertEqual(decimal2minifloat(.01367188),
                         7,
                         'failed to round subnormal float')
        self.assertEqual(decimal2minifloat([.0039]),
                         2,
                         'failed to round subnormal list')
        self.assertEqual(decimal2minifloat([-500, 490]), [255, 127], 'failed to deal with extremes')

        # Negative zero (128 in minifloat) is not used and therefore not tested
        integer_values = [x for x in range(128)]
        integer_results = decimal2minifloat(minifloat2decimal(integer_values))
        self.assertEqual(integer_results,
                         integer_values,
                         'failed to convert positive range')

        integer_values = [x for x in range(129, 255)]
        integer_results = decimal2minifloat(minifloat2decimal(integer_values))
        self.assertEqual(integer_results,
                         integer_values,
                         'failed to convert negative range')


if __name__ == '__main__':
    unittest.main(verbosity=2, exit=False)
