import pandas as pd
from elephant.statistics import isi, cv
import numpy as np

spk = pd.read_csv('bal_stdp_cudadgx3/output_spikes.csv',
                  dtype={"time_ms": float, "id": int})

n_exc = 90000

min_ms = np.floor(np.amin(spk["time_ms"]))
max_ms = np.ceil(np.amax(spk["time_ms"]))

rate, _ = np.histogram(spk["id"], bins=range(n_exc + 1))
assert len(rate) == n_exc
mean_rate = np.divide(rate, (max_ms - min_ms) / 1000.0, dtype=float)
print("Mean firing rate: %fHz" % np.average(mean_rate))

# Sort spikes by id
neuron_spikes = spk.groupby("id")

# Loop through neurons
cv_isi = []
for n in range(n_exc):
    try:
        # Get this neuron's spike times
        neuron_spike_times = neuron_spikes.get_group(n)["time_ms"].values

        # If this neuron spiked more than once i.e. it is possible to calculate ISI!
        if len(neuron_spike_times) > 1:
            cv_isi.append(cv(isi(neuron_spike_times)))
    except KeyError:
        pass

print("Mean CV ISI: %f" % np.average(cv_isi))

bin_spk_t = None
for n in np.random.choice(n_exc, 1000, replace=False):
    #print(f'Selected neuron {n}')
    spk_t = spk.loc[spk['id']==n].time_ms
    
    temp_count, _ = np.histogram(spk_t, bins=np.arange(min_ms, max_ms, 3))
    if bin_spk_t is None:
        bin_spk_t = temp_count
    else:
        bin_spk_t += temp_count
    #print(f'Intermediary array {bin_spk_t}')

#print(f'Final array {bin_spk_t}')
mean_spk_count = np.average(bin_spk_t)
#print(f'mean {mean_spk_count}')
var_spk_count = np.var(bin_spk_t)
#print(f'variance {var_spk_count}')
print(f'Fano factor: {var_spk_count/mean_spk_count}')
