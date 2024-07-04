from simulations.liquid_state_machine import liquid_state_machine
from simulations.sleep_normalization import sleep_normalization
from simulations.fp8_potjans_diesmann import fp8_potjans_diesmann
from simulations.neuron_synapse_models import neuron_synapse_models
from simulations.stdp import stdp
from simulations.balanced_network import balanced_network
from simulations.balanced_network_stdp import balanced_network_stdp
from simulations.minifloat import minifloat_operations
from simulations.istdp import istdp

import os
from datetime import datetime

import argparse
from brian2 import DEFAULT_FUNCTIONS, prefs, set_device, ms

from core.utils.misc import stochastic_decay, fp8_multiply, fp8_add,\
    fp8_add_stochastic, fp8_multiply_stochastic, fp8_smaller_than, deterministic_decay
DEFAULT_FUNCTIONS.update({'stochastic_decay': stochastic_decay,
                          'fp8_multiply': fp8_multiply,
                          'fp8_multiply_stochastic': fp8_multiply_stochastic,
                          'fp8_add': fp8_add,
                          'fp8_add_stochastic': fp8_add_stochastic,
                          'fp8_smaller_than': fp8_smaller_than,
                          'deterministic_decay': deterministic_decay})

parser = argparse.ArgumentParser(
    description='Main file that executes specified simulation',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--save_path',
                    type=str,
                    default=f"""{datetime.now().strftime('%Y.%m.%d')}"""
                            f"""_{datetime.now().hour}."""
                            f"""{datetime.now().minute}/""",
                    help=f'directory to save, creating folder and overwriting '
                         f'existing ones. Defaults to datetime name.')
parser.add_argument('--code_path',
                    type=str,
                    default='./code/',
                    help='directory of compiled code')
parser.add_argument('--quiet',
                    action='store_true',
                    help='whether to create plots/info or not')
parser.add_argument('--backend',
                    type=str,
                    default='numpy',
                    help='Backend used by brian2 to run simulation')
parser.add_argument('--timestep',
                    default=1,
                    type=float,
                    help='timestep of the simulation (dt), in ms')

subparsers = parser.add_subparsers(title='Simulation files available',
                                   help='Additional help for each available')

subparser_lsm = subparsers.add_parser(
    'LSM',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
subparser_lsm.add_argument('--size',
                           type=int,
                           default=128,
                           help='size of the liquid')
subparser_lsm.add_argument('--trial',
                           type=int,
                           default=0,
                           help='trial number')
subparser_lsm.add_argument('--precision',
                           type=str,
                           default='fp64',
                           help=f'Bit precision used. Currently only supports '
                                f'8 and 64')
subparser_lsm.set_defaults(func=liquid_state_machine)

subparser_sleep = subparsers.add_parser(
    'sleep',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
subparser_sleep.add_argument('--size',
                             type=int,
                             default=128,
                             help='size of the liquid')
subparser_sleep.add_argument('--trial',
                             type=int,
                             default=0,
                             help='trial number')
subparser_sleep.add_argument('--precision',
                             type=str,
                             default='fp64',
                             help=f'Bit precision used. Currently only '
                                  f'supports 8 and 64')
subparser_sleep.set_defaults(func=sleep_normalization)

subparser_pd = subparsers.add_parser(
    'PD',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
subparser_pd.add_argument('--protocol',
                          type=int,
                          help=f'Type of stimulation. 1 is for spontaneous '
                               f'input whereas 2 is for thalamic')
subparser_pd.add_argument('--w_in',
                          type=float,
                          help=f'Relative strength of weights.')
subparser_pd.add_argument('--bg_freq',
                          type=float,
                          help=f'Strength of background noise, in Hz.')
subparser_pd.add_argument('--rounding',
                          type=str,
                          default='stochastic',
                          help=f'Rounding scheme. Nearest or stochastic.')
subparser_pd.set_defaults(func=fp8_potjans_diesmann)

subparser_models = subparsers.add_parser(
    'models',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
subparser_models.set_defaults(func=neuron_synapse_models)

subparser_balance = subparsers.add_parser(
    'balance',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
subparser_balance.add_argument('--w_perc',
                               type=float,
                               help=f'Relative strength of weights. In the '
                                    f'case of minifloat, it represents '
                                    f'decimal value.')
subparser_balance.add_argument('--trial',
                               type=int,
                               default=0,
                               help='trial number')
subparser_balance.set_defaults(func=balanced_network)

subparser_stdp = subparsers.add_parser(
    'STDP',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
subparser_stdp.add_argument('--protocol',
                            type=int,
                            default=1,
                            help=f'Type of simulation. 1 is for '
                                 f'general weight changes over time '
                                 f'and 2 is for kernel.')
subparser_stdp.add_argument('--N_post',
                            type=int,
                            default=1,
                            help=f'Number of postsynaptic neurons. This '
                                 f'is used in protocol 3 only.')
subparser_stdp.add_argument('--tmax',
                            type=int,
                            default=100000,
                            help=f'Simulation time. This is used in protocol '
                                 f'3 only.')
subparser_stdp.add_argument('--precision',
                            type=str,
                            default='fp64',
                            help=f'Bit precision used. Currently only supports '
                                 f'8 and 64.')
subparser_stdp.add_argument('--w_max',
                            type=float,
                            default=100,
                            help=f'Maximum weight value, in mV. Used '
                                 f'only in protocol 3.')
subparser_stdp.add_argument('--event_condition',
                            type=str,
                            default= 'abs(Ca) > 0.01',
                            help=f'Condition uppon a plasticity event is '
                                 f'triggered.')
subparser_stdp.add_argument('--w_init',
                            type=float,
                            default=11,
                            help=f'Initial value of weights. To be used only with '
                                 f'protocol 1 that uses fp8.')
subparser_stdp.set_defaults(func=stdp)

subparser_balance_stdp = subparsers.add_parser(
    'balance_stdp',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
subparser_balance_stdp.add_argument('--precision',
                                    type=str,
                                    default='fp64',
                                    help=f'Bit precision used. Currently only supports '
                                         f'8 and 64')
subparser_balance_stdp.add_argument('--w_max',
                                    type=float,
                                    default=1000,
                                    help='Maximum value of plastic weight, in mV')
subparser_balance_stdp.add_argument('--event_condition',
                                    type=str,
                                    default= 'abs(Ca) > 0',
                                    help=f'Condition uppon a plasticity event is '
                                         f'triggered.')
subparser_balance_stdp.add_argument('--protocol',
                                    type=int,
                                    default=1,
                                    help=f'Type of simulation. 1 is for '
                                         f'bimodal-like distribution '
                                         f'and 2 is for unimodal.')
subparser_balance_stdp.add_argument('--we',
                                    type=float,
                                    default=1,
                                    help=f'Initial excitatory weight.')
subparser_balance_stdp.add_argument('--alpha',
                                    type=float,
                                    default=0.1449,
                                    help=f'Depression factor for protocol 2.')
subparser_balance_stdp.add_argument('--tsim',
                                    type=int,
                                    default=200,
                                    help=f'Simulation time.')
subparser_balance_stdp.add_argument('--ca_decays',
                                    type=str,
                                    default='20*ms',
                                    help=f'Decay values of plasticity windows.')
subparser_balance_stdp.set_defaults(func=balanced_network_stdp)

subparser_minifloat = subparsers.add_parser(
    'minifloat',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
subparser_minifloat.add_argument('--protocol',
                                 type=int,
                                 help=f'Type of simulation. 1 is for '
                                      f'multiplication and 2 is for addition.'
                                      f' All operations are stochastic.')
subparser_minifloat.set_defaults(func=minifloat_operations)

subparser_istdp = subparsers.add_parser(
    'iSTDP',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
subparser_istdp.add_argument('--event_condition',
                             type=str,
                             default= 'Ca > 0',
                             help=f'Condition uppon a plasticity event is '
                                  f'triggered.')
subparser_istdp.add_argument('--protocol',
                             type=int,
                             help=f'Type of simulation. 1 and 2 are for high '
                                  f'and low learning rates, respectively.')
subparser_istdp.set_defaults(func=istdp)

args = parser.parse_args()

os.makedirs(args.save_path, exist_ok=True)

if args.backend == 'numpy':
    prefs.codegen.target = args.backend
elif args.backend == 'cpp_standalone':
    set_device('cpp_standalone', build_on_run=False)
elif args.backend == 'cuda_standalone':
    import brian2cuda
    set_device('cuda_standalone', build_on_run=False)
elif args.backend == 'markdown':
    set_device(args.backend, filename='model_description')
else:
    raise UserWarning('Backend not supported')

args.func(args)
