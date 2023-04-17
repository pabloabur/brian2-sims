from core.utils.misc import minifloat2decimal, fp8_add, fp8_multiply
import numpy as np
import plotext as plt


def minifloat_operations(num1, num2, operation):
    """
    Useful when analysing and plotting minifloats and their arithmetic
    operations

    Parameters
    ----------
    num1, num2 : int
        Operands.
    operation : str
        'mul' or 'add'

    Examples
    --------
    >>> from core.utils.minifloat import minifloat_operations
    >>> subnrm = 7
    >>> nrm = [i for i in range(1, 8)]
    >>> [minifloat_operations(n, subnrm, 'mul') for n in nrm]
    """
    if operation == 'mul':
        c_c = fp8_multiply(num1, num2, 1)
    if operation == 'add':
        c_c = fp8_add(num1, num2, 1)

    c_next = c_c + 1
    c_prev = c_c - 1 if c_c > 0 else 129
    c_values = [minifloat2decimal([c_prev])[0],
                minifloat2decimal([c_c])[0],
                minifloat2decimal([c_next])[0]]

    c_dec = minifloat2decimal([c_c])
    a_dec = minifloat2decimal(num1)
    b_dec = minifloat2decimal(num2)
    if operation == 'mul':
        c_i = a_dec*b_dec
        print(f'Ideal result is {a_dec} * {b_dec} = {c_i}')
    if operation == 'add':
        c_i = a_dec + b_dec
        print(f'Ideal result is {a_dec} + {b_dec} = {c_i}')
    print(f'Calculated result was {c_dec}')
    print(f'Absolute error was {np.abs(c_i-c_dec)}')

    plt.clear_figure()
    plt.theme('clear')
    plt.plot_size(height=5, width=50)
    plt.scatter(c_values, [0 for _ in c_values])
    plt.scatter(c_i, [0], color='red+')
    plt.show()
    print('---------------------------------------')
