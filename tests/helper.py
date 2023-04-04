from core.utils.misc import *
import numpy as np
import plotext as plt
import pdb

plt.theme('clear')
plt.plot_size(height=5, width=50)

def helper(a, b):
    if a==7 and b==25:
        pdb.set_trace()
    c_c = fp8_multiply(a, b, 1)

    c_next = c_c + 1
    c_prev = c_c - 1
    dist = np.abs(minifloat2decimal([c_next])-minifloat2decimal([c_prev]))
    please = [minifloat2decimal([c_prev])[0], minifloat2decimal([c_c])[0], minifloat2decimal([c_next])[0]]

    c_dec = minifloat2decimal([c_c])
    a_dec = minifloat2decimal(a)
    b_dec = minifloat2decimal(b)
    c_i = a_dec*b_dec
    print(f'Ideal result is {a_dec} * {b_dec} = {c_i}')
    print(f'Calculated result was {c_dec}')
    print(f'Absolute error was {np.abs(c_i-c_dec)}')

    plt.scatter(please, [0 for _ in please])
    plt.scatter(c_i, [0], color='red+')
    plt.show()
    print('---------------------------------------')

## Subnormals and subnormals
# Always correclty rounds to zero
# subnrm = [7]
# nrm = [i for i in range(1, 8)]
# for sb in subnrm:
#     for n in nrm:
#         helper(sb, n)

## Subnormals and normals
# Biggest subnormal manages to bring about cancellation up until normal 25
subnrm = [7]
nrm = [i for i in range(8, 26)]
for sb in subnrm:
    for n in nrm:
        print(f'minifloat {sb} and {n}')
        helper(sb, n)
