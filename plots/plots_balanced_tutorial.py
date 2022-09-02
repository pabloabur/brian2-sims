import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pickle
import json
from teili.tools.misc import minifloat2decimal

if len(sys.argv) < 2:
    raise Exception(f'Provide directory where data was saved')
root_dir = Path(sys.argv[1])

rate_means, rate_std, wi = [], [], []

files = sorted(root_dir.glob('*'))
for file_desc in files:
    if file_desc.is_file():
        continue
    try:
        rates = np.load(file_desc / 'avg_rate.npy')
    except FileNotFoundError:
        continue
    rate_means.append(np.mean(rates))
    rate_std.append(np.std(rates) / np.sqrt(len(rates)))

    with open(file_desc / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    wi.append(minifloat2decimal(metadata['inh_weight']))

plt.figure()
plt.errorbar(wi, rate_means, rate_std, capsize=10)
plt.ylabel('rate (Hz)')
plt.xlabel('inhibitory weight strength')
plt.show()
