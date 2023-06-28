from core.utils.minifloat import *
from core.utils.misc import decimal2minifloat, fp8_add
import numpy as np
import plotext as plt
import matplotlib.pyplot as mpl

ca_trace = [127]
for _ in range(50): ca_trace.append(fp8_multiply(ca_trace[-1], 55, 1))
plt.clear_figure()
plt.plot(minifloat2decimal(ca_trace))
plt.show()

eta = 20
# scenario 1
w0=[100 for _ in range(len(ca_trace))]
# scenario 2
#w0=[88 for _ in range(len(ca_trace))]
cum_res = np.array([0 for _ in range(len(ca_trace))])
delta_w = [fp8_multiply(eta, x, 1) for x in ca_trace]
print(delta_w)
N = 500
for i in range(N):
    trial_res = []
    for w, dw in zip(w0, delta_w): trial_res.append(fp8_add_stochastic(w, dw, 1))
    cum_res = cum_res + minifloat2decimal(trial_res)

cum_res = cum_res / N

plt.clear_figure()
plt.plot(minifloat2decimal(trial_res))
plt.plot(cum_res)
plt.show()
mpl.plot(minifloat2decimal(trial_res)[:30], label='single trial')
mpl.plot(cum_res[:30], label='average')
mpl.legend()
mpl.savefig('sim_data/w011round_det_inceta.png')
