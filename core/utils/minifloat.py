from core.utils.misc import *
import numpy as np
import plotext as plt

def minifloat_operations(num1, num2):
    """
    Useful when analysing and plotting minifloats and their arithmetic operations

    Parameters
    ----------
    num1, num2 : int
        Operands.

    Examples
    --------
    >>> subnrm = [7]
    >>> nrm = [i for i in range(1, 8)]
    >>> [minifloat_operations(n, subnrm) for n in nrm]
    """
    c_c = fp8_multiply(num1, num2, 1)

    c_next = c_c + 1
    c_prev = c_c - 1 if c_c>0 else 129
    dist = np.abs(minifloat2decimal([c_next])-minifloat2decimal([c_prev]))
    please = [minifloat2decimal([c_prev])[0], minifloat2decimal([c_c])[0], minifloat2decimal([c_next])[0]]

    c_dec = minifloat2decimal([c_c])
    a_dec = minifloat2decimal(num1)
    b_dec = minifloat2decimal(num2)
    c_i = a_dec*b_dec
    print(f'Ideal result is {a_dec} * {b_dec} = {c_i}')
    print(f'Calculated result was {c_dec}')
    print(f'Absolute error was {np.abs(c_i-c_dec)}')

    plt.clear_figure()
    plt.theme('clear')
    plt.plot_size(height=5, width=50)
    plt.scatter(please, [0 for _ in please])
    plt.scatter(c_i, [0], color='red+')
    plt.show()
    print('---------------------------------------')
