import os
from datetime import datetime

import git
import sys
sys.path.extend([git.Repo('.').git.rev_parse('--show-toplevel')])

import argparse
from brian2 import defaultclock, prefs, set_device
from brian2 import ms

parser = argparse.ArgumentParser(description='Main simulation file')
parser.add_argument('--sim', type=str, help='Simulation file to run')
parser.add_argument('--precision', type=str, default='fp64',
                    help='Bit precision used. Currently only supports 8 and 64')
parser.add_argument('--trial', type=int, default=0, help='trial number')
parser.add_argument('--save_path', default=None, help='directory to save')
parser.add_argument('--code_path', default=None, help='directory of compiled code')
parser.add_argument('--quiet', action='store_true',
                    help='whether to create plots/info or not')
parser.add_argument('--backend', type=str, default='numpy',
                    help='Backend used by brian2 to run simulation')
# In case of LSM
parser.add_argument('--size', type=int, default=128, help='size of the liquid')
# In case of P&D
parser.add_argument('--protocol', type=int,
                    help=f'Type of stimulation. 1 is for spontaneous input '
                         f'whereas 2 is for thalamic')
parser.add_argument('--w_in', type=float, help=f'Relative strength of '
                                               f'weights.')
parser.add_argument('--bg_freq', type=float, help=f'Strength of background '
                                                  f'noise.')
args = parser.parse_args()
sim = args.sim
backend = args.backend
precision = args.precision
trial_no = args.trial
save_path = args.save_path
code_path = args.code_path
quiet = args.quiet
protocol = args.protocol

size = args.size

if protocol==1:
    w_in = 50
    bg_freq = 74
elif protocol==2:
    w_in = args.w_in
    bg_freq = args.bg_freq
else:
    raise UserWarning('A protocol must be chosen for P&D simulation')

defaultclock.dt = 1*ms

if not save_path:
    date_time = datetime.now()
    save_path = f"""{date_time.strftime('%Y.%m.%d')}_{date_time.hour}.{date_time.minute}/"""
os.makedirs(save_path, exist_ok=True)

if not code_path:
    code_path = './code/'

if backend == 'numpy':
    prefs.codegen.target = backend
elif backend == 'cpp_standalone':
    set_device(backend, build_on_run=False)
elif backend == 'markdown':
    set_device('markdown', filename='model_description')
else:
    raise UserWarning('Backend not supported')

if sim=='LSM':
    from simulations.liquid_state_machine import liquid_state_machine

    liquid_state_machine(size, precision, defaultclock, trial_no,
                         save_path, code_path, quiet)
elif sim=='PD':
    from simulations.fp8_potjans_diesmann import fp8_potjans_diesmann
    # default value for background rate was not 8 as in original paper because
    # I am using a slightly different way of generating poisson input
    if protocol==1:
        fp8_potjans_diesmann(protocol, bg_freq, w_in, defaultclock,
                         save_path, code_path)
    else:
        fp8_potjans_diesmann(protocol, defaultclock=defaultclock,
                             save_path=save_path, code_path=code_path)
