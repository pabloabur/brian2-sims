import os
from datetime import datetime

import argparse
from brian2 import defaultclock, prefs, set_device
from brian2 import ms

from simulations.liquid_state_machine import liquid_state_machine

parser = argparse.ArgumentParser(description='LSM with distinct resolutions')
parser.add_argument('--trial', type=int, default=0, help='trial number')
parser.add_argument('--path', default=None, help='directory to save')
parser.add_argument('--quiet', action='store_true',
                    help='whether to create plots/info or not')
parser.add_argument('--backend', type=str, default='numpy',
                    help='Backend used by brian2 to run simulation')
args = parser.parse_args()
backend = args.backend
trial_no = args.trial
path = args.path
quiet = args.quiet

defaultclock.dt = 1*ms

if backend == 'numpy':
    prefs.codegen.target = backend
elif backend == 'cpp_standalone':
    set_device(backend, build_on_run=False)
elif backend == 'markdown':
    set_device('markdown', filename='model_description')
else:
    raise UserWarning('Backend not supported')

if not path:
    date_time = datetime.now()
    path = f"""{date_time.strftime('%Y.%m.%d')}_{date_time.hour}.{date_time.minute}/"""
os.makedirs(path)

liquid_state_machine(defaultclock, trial_no, path, quiet)
