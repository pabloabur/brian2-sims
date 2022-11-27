from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

win, bg, ai = [], [], []
for fn in Path('.').glob('*.csv'):
    dat = pd.read_csv(fn)
    bg.append(float(fn.name.split('_')[2][2:]))
    win.append(float(fn.name.split('_')[3][3:5]))
    ai.append(100*sum(dat['AI'])/8)

x, y = np.unique(win), np.unique(bg)
z = np.zeros((len(x), len(y)))
for i, v in enumerate(ai):
    z[np.where(y==bg[i])[0][0], np.where(x==win[i])[0][0]] = v
x, y = np.meshgrid(x, y)
levels = np.linspace(0, 100, 10)
CS = plt.contourf(x, y, z, levels=levels)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.colorbar(CS)
plt.show()
