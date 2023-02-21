library(jsonlite)
library(argparser)
include('plots/parse_inputs.R')

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

files = Path('.').glob('*.txt')

acc, size, prec = [], [], []
for fn in files:
    with open(fn, 'r') as f:
        acc.append(float(f.readline()))
    size.append(int(fn.name.split('-')[0][5:]))
    prec.append(int(fn.name.split('-')[1][5:]))

acc = np.array(acc)
size = np.array(size)
prec = np.array(prec)
x = np.unique(size)

mean_fp8, e_fp8, mean_fp64, e_fp64 = [], [], [], []
for s in x:
    mean_fp8.append(np.mean(acc[(size==s) & (prec==8)]))
    e_fp8.append(np.std(acc[(size==s) & (prec==8)]) / np.sqrt(10))
    mean_fp64.append(np.mean(acc[(size==s) & (prec==64)]))
    e_fp64.append(np.std(acc[(size==s) & (prec==64)]) / np.sqrt(10))

plt.plot(x, mean_fp64, label='full precision', linewidth=3, color='k')
plt.plot(x, mean_fp8, label='minifloat', linewidth=3, color='r')
plt.errorbar(x, mean_fp8, e_fp8, color='r', linestyle='None')
plt.errorbar(x, mean_fp64, e_fp64, color='k', linestyle='None')
plt.xticks(fontsize=14)
plt.yticks(fontsize=16)
plt.legend()
plt.show()
