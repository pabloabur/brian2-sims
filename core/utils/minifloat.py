from core.utils.misc import minifloat2decimal, fp8_add_stochastic, fp8_add,\
    fp8_multiply, fp8_add_stochastic_debug
import numpy as np
import plotext as plt

def minifloat_operations(num1, num2, operation, stochastic=False):
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
        result = fp8_multiply(num1, num2, 1)
    if operation == 'add':
        if stochastic:
            result, disc_bits, rand_num, rand_num_len, delta = fp8_add_stochastic_debug(num1, num2, 1)
        else:
            result = fp8_add(num1, num2, 1)

    res_next = result + 1
    res_prev = result - 1 if result > 0 else 129
    values = [minifloat2decimal([res_prev])[0],
                minifloat2decimal([result])[0],
                minifloat2decimal([res_next])[0]]

    a_dec = minifloat2decimal(num1)
    b_dec = minifloat2decimal(num2)
    if operation == 'mul':
        res_ideal = a_dec*b_dec
        print(f'Ideal result is {a_dec} * {b_dec} = {res_ideal}')
    if operation == 'add':
        res_ideal = a_dec + b_dec
        print(f'Ideal result is {a_dec} + {b_dec} = {res_ideal}')
    print(f'Calculated result was {values[1]}')
    print(f'Absolute error was {np.abs(res_ideal-values[1])}')

    values = sorted(values)
    if values[1]==res_ideal:
        distance_to_lower = 0
        distance_to_higher = 0
        ideal_prob = 0
    else:
        lower = None
        higher = None
        for val in values:
            if val <= res_ideal:
                lower = val
            if val >= res_ideal and higher is None:
                higher = val
        distance_to_lower = abs(res_ideal - lower)
        distance_to_higher = abs(higher - res_ideal)
        ideal_prob = distance_to_higher / (distance_to_lower + distance_to_higher)

    if distance_to_lower < distance_to_higher:
        most_likely = res_ideal - distance_to_lower
    elif distance_to_higher < distance_to_lower:
        most_likely = res_ideal + distance_to_higher
    # Arbitrarily takes central value. Works for exact and 50/50 cases
    else:
        most_likely = values[1]

    # TODO all cases and multiples trials like this
    #ws = [135, 232, 238, 68, 145, 39, 31, 22, 14, 32, 56, 56, 48]
    #g0 = [8, 239, 239, 44, 56, 15, 15, 15, 23, 33, 88, 112, 120]
    #for _ in range(trials):
    #    minifloat_operations(239, 135, 'add', True)
    # TODO fix calculations here. Use returned values
    # TODO show delta, probability; maybe another script with many runs to get average
    # TODO Show how many times it went to most likely?
    lfsr_len = 3
    low_prob = delta - 6
    if low_prob>0:
        aux_disc_bits = 1 if disc_bits==0 else disc_bits
        calc_prob = (aux_disc_bits/2**lfsr_len)*(1/2**low_prob)
    else:
        calc_prob = (2**lfsr_len - disc_bits) / (2**lfsr_len)
    print(f'Rounding down probability: {ideal_prob}')
    print(f'Calculated probability: {calc_prob}')

    plt.clear_figure()
    plt.theme('clear')
    plt.plot_size(height=5, width=50)
    plt.scatter(values, [0 for _ in values])
    plt.scatter(res_ideal, [0], color='red+')
    plt.show()
    print('---------------------------------------')
