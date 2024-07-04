import csv
import matplotlib.pyplot as plt
import numpy as np

spks = []
with open('../datasets/spikes.csv', newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        spks.append(row)

speaker = [int(x[0]) for x in spks]
digit = [int(x[1]) for x in spks]
times = [[float(y) for y in x[2:]] for x in spks]

sp1 = np.where(speaker==1)[0]
sp1_dg1 = np.where(digit[sp1]==1)[0]
print(sp1_dg1)
