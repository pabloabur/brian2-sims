import sys
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
import pickle
import json

from core.utils.misc import minifloat2decimal

if len(sys.argv) < 2:
    raise Exception(f'Provide directory where data was saved')
data_folder = Path(sys.argv[1])

experiments = sorted(data_folder.glob('**/metadata.json'))
if not experiments:
    experiments = ['']

wis, trials = [], []
for exp in experiments:
    with open(exp, 'r') as f:
        desc = json.load(f)
        wis.append(desc['inh_weight'])
        trials.append(desc['trial_no'])
wis = list(OrderedDict.fromkeys(wis))
trials = list(OrderedDict.fromkeys(trials))

rate_means = np.zeros((len(wis), len(trials)))
rate_means.fill(np.nan)
rate_std = np.zeros((len(wis), len(trials)))
rate_std.fill(np.nan)

experiments = sorted(data_folder.glob('*'))

# experiments with multiple trials are stored in folders, but this script can
# also plot metrics for a single trial
removed_folders = []
for exp in experiments:
    if exp.is_dir():
        if len(list(exp.glob('*'))) <= 1:
            removed_folders.append(exp)
experiments = [exp for exp in experiments if exp.name != 'description.txt']
for rm in removed_folders: experiments.remove(rm)

for exp in experiments:
    if exp.is_file():
        exp = exp.parent
    with open(exp / 'metadata.json', 'r') as f:
        metadata = json.load(f)

    rates = np.load(exp / 'rates.npz')
    row_index = wis.index(metadata['inh_weight'])
    col_index = trials.index(metadata['trial_no'])
    rate_means[row_index, col_index] = np.mean(rates['rates'])

plt.figure()
plt.errorbar(wis, np.mean(rate_means, axis=1),
             np.std(rate_means, axis=1) / np.sqrt(len(trials)),
             capsize=10)
plt.ylabel('rate (Hz)')
plt.xlabel('inhibitory weight strength')
plt.ylim([0, np.max(rate_means)])
plt.show()
