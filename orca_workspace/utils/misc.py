from brian2 import implementation, check_units, ms, declare_types,\
        SpikeMonitor, Network, NeuronGroup, TimedArray, Function,\
        DEFAULT_FUNCTIONS
import numpy as np

# Parameters for 8-bit floating point implementation
EXP_WIDTH = 4
FRAC_WIDTH = 3
SIGN_WIDTH = 1
N_BITS = SIGN_WIDTH + EXP_WIDTH + FRAC_WIDTH
REPR_MASK = 2**N_BITS - 1
FRAC_MASK = 2**FRAC_WIDTH - 1
GUARD_WIDTH = 3
# Smallest normal: 2^-(-BIAS+1)
# Smallest subnormal: 2^(-BIAS+1)/2^(FRAC_WIDTH-1)
# Biased representation of exponents, i.e. what is actually stored in hardware
EMIN = 0
EMAX = 15
BIAS = 7


def neuron_group_from_spikes(num_inputs, simulation_dt, duration,
                             poisson_group=None, spike_indices=None,
                             spike_times=None):
    """Converts spike activity in a neuron poisson_group with the same
    activity.

    Args:
        num_inputs (int): Number of input channels from source.
        simulation_dt (brian2.unit.ms): Time step of simulation.
        duration (int): Duration of simulation in brian2.ms.
        poisson_group (brian2.poissonGroup): Poisson poisson_group that is
            passed instead of spike times and indices.
        spike_indices (numpy.array): Indices of the original source.
        spike_times (numpy.array): Time stamps with unit of original spikes in
            ms.

    Returns:
        neu_group (brian2 object): Neuron poisson_group with mimicked activity.
    """
    if spike_indices is None and spike_indices is None:
        net = Network()
        try:
            monitor = SpikeMonitor(poisson_group)
        except TypeError:
            raise
        net.add(poisson_group, monitor)
        net.run(duration)
        spike_times, spike_indices = monitor.t, monitor.i
    spike_times = [spike_times[np.where(spike_indices == i)[0]]
                   for i in range(num_inputs)]
    # Create matrix where each row (neuron id) is associated with time when
    # there is a spike or -1 when there is not
    converted_input = (np.zeros((num_inputs,
                                 np.around(duration/simulation_dt).astype(int)+1
                                 ))
                       - 1)*simulation_dt
    for ind, val in enumerate(spike_times):
        # Prevents floating point errors
        int_values = np.around(val/simulation_dt).astype(int)

        converted_input[ind, int_values] = int_values * simulation_dt
    converted_input = np.transpose(converted_input)
    converted_input = TimedArray(converted_input, dt=simulation_dt)
    # t is simulation time, and will be equal to tspike when there is a spike.
    # Cell remains refractory when there is no spike, i.e. when tspike=-1
    neu_group = NeuronGroup(num_inputs,
                            model='tspike=converted_input(t, i): second',
                            threshold='t==tspike',
                            refractory='tspike < 0*ms')
    neu_group.namespace.update({'converted_input': converted_input})

    return neu_group

def stochastic_decay(init_value, decay_numerator, rand_num_bits, _vectorisation_idx):
    """ This function implements an stochastic exponential decay suited
        for digitial hardware implementations. It is mathematically
        described as V = int(V*tau/(tau + dt) + rand())
    Args:
        init_value (list or numpy.array): Values to be decayed
        decay_numerator (int): Values that when divided by 256
            represents decay rate, that is tau/(tau + dt)
        rand_num_bits (int): Number of bits of random number generator.
    """
    rand_num_bits = int(rand_num_bits)
    init_value_sign = np.sign(init_value)
    init_value =  np.abs(init_value.astype(int))
    lfsr = np.ceil(np.random.rand(len(init_value))*(2**rand_num_bits-1)).astype(int)
    # Using + instead of | add lfsr to rand_num_bits, which is not what
    # we want. If the former is used, parenthesis must indicate order.
    new_val = init_value<<rand_num_bits | lfsr
    # Decay is done by dividing by an 8-bit number
    new_val = (new_val*decay_numerator.astype(int)) >> 8
    new_val = new_val >> int(rand_num_bits)
    return new_val * init_value_sign
stochastic_decay = Function(stochastic_decay, arg_units=[1, 1, 1],
                            return_unit=1, stateless=False,
                            auto_vectorise=True)
cpp_code = '''
int stochastic_decay(int init_value, int decay_numerator, int rand_num_bits, int _vectorisation_idx)
{
    int new_val;
    int values;
    int lfsr;

    values = 0;
    while(rand_num_bits!=0){
        values *= 2;
        --rand_num_bits;
    }
    values -= 1;
    lfsr = ceil(rand(_vectorisation_idx) * values);
    new_val = init_value<<rand_num_bits | lfsr;
    new_val = (new_val*decay_numerator) >> 8;
    new_val = new_val >> rand_num_bits;
    return new_val;
}
'''
stochastic_decay.implementations.add_implementation('cpp', cpp_code,
    dependencies={'rand': DEFAULT_FUNCTIONS['rand'],
                  'ceil': DEFAULT_FUNCTIONS['ceil']})

def deterministic_decay(init_value, decay_numerator):
    """ This function implements an exponential decay suited
        for digitial hardware implementations. It is mathematically
        described as V = V*tau/(tau + dt))
    Args:
        init_value (list or numpy.array): Values to be decayed
        decay_numerator (int): Values that when divided by 256
            represents decay rate, that is tau/(tau + dt)
    """
    init_value_sign = np.sign(init_value)
    init_value =  np.abs(init_value.astype(int))
    new_val = (init_value*decay_numerator.astype(int)) >> 8
    return new_val * init_value_sign
deterministic_decay = Function(deterministic_decay, arg_units=[1, 1],
                            return_unit=1, stateless=False)


def extract_fields(bitstring):
    """ This function follows some of the IEEE 754 standard for
        floating point representation.

    Args:
        bitstring (list or numpy.array): Must be equivalent to an binary word.
    Returns:
        numpy.array: sign, exponent, absolute value, and normal flag, in this
        order
    """
    sign = bitstring >> (EXP_WIDTH+FRAC_WIDTH)
    abs_val = bitstring & 0x7F  # Set sign bit to 0
    is_normal = abs_val >= 2 ** FRAC_WIDTH
    exponent = abs_val >> FRAC_WIDTH
    return sign, exponent, abs_val, is_normal


def get_leading_zeros(bitstring):
    """ This function follows some of the IEEE 754 standard for
        floating point representation. It assumes an 8-bit
        binary word.

    Args:
        bitstring (list or numpy.array): Binary word.
    Returns:
        leading_zeros (list or numpy.array): Number of zeros left of input
            word.
    """
    leading_zeros = np.zeros_like(bitstring)
    bitstring = np.array(bitstring)
    leading_zeros[bitstring == 0] = N_BITS
    # Ensures bitstrings above are left untouched
    bitstring[bitstring == 0] = REPR_MASK

    leading_zeros = np.where(bitstring <= 0x0F,
                             leading_zeros + 4,
                             leading_zeros)
    bitstring[bitstring <= 0x0F] <<= 4

    leading_zeros = np.where(bitstring <= 0x3F,
                             leading_zeros + 2,
                             leading_zeros)
    bitstring[bitstring <= 0x3F] <<= 2

    leading_zeros = np.where(bitstring <= 0x7F,
                             leading_zeros + 1,
                             leading_zeros)
    bitstring[bitstring <= 0x7F] <<= 1

    return leading_zeros

def decimal2minifloat(decimal):
    """ Converts a representable decimal value to 8-bit floating point.
        Use it with CAUTION as it does not check if decimal provided is
        actually supported in the implemented format.

    Args:
        decimal (float): decimal value to be converted
    Returns:
        flot_repr (ndarray): floating point representation of decimal.
    """
    if decimal == 0:
        return 0

    sign = 0 if decimal >= 0 else 1
    decimal_abs = np.abs(decimal)
    int_part = int(str(decimal_abs).split('.')[0])
    frac_part = decimal_abs - int_part
    
    int_bits = []
    remainder = int_part
    while remainder != 0:
        int_bits.append(int(remainder % 2))
        remainder = int(remainder / 2)
    # For convenience order is switched, that is MSB is in LSB position

    frac_bits = []
    remainder = frac_part
    while remainder != 0:
        frac_bits.append(int(remainder * 2))
        remainder = remainder*2 - int(remainder * 2)

    normal_bit_position = 0
    if not int_bits:
        while True:
            if frac_bits[normal_bit_position] == 1:
                normal_bit_position += 1
                break
            normal_bit_position += 1
        if normal_bit_position > np.abs(EMIN-BIAS+1):
            normal_bit_position = np.abs(EMIN-BIAS+1)
        normal_bit = frac_bits[normal_bit_position - 1]
        # Note that slicing out of bounds values does not return anything
        upper_limit = normal_bit_position + FRAC_WIDTH
        while upper_limit>len(frac_bits):
            upper_limit -= 1
        frac_part = frac_bits[normal_bit_position:upper_limit]
        frac_part.reverse()
        while len(frac_part) < FRAC_WIDTH:
            frac_part = [0] + frac_part
        exp = -normal_bit_position
    else:
        normal_bit = int_bits[-1]
        normal_bit_position = len(int_bits) - 1
        if normal_bit_position > EMAX-BIAS:
            normal_bit_position = EMAX-BIAS
        exp = normal_bit_position

        frac_bits = frac_bits[::-1] + int_bits[:-1]
        lower_limit = len(frac_bits) - FRAC_WIDTH
        if lower_limit < 0:
            for _ in range(np.abs(lower_limit)):
                frac_bits = [0] + frac_bits
                lower_limit += 1
        frac_part = frac_bits[lower_limit:]

    exp += (BIAS-1)
    exp += normal_bit

    cumm_frac = 0
    for idx in range(FRAC_WIDTH):
        cumm_frac += frac_part[idx]*2**idx

    minifloat_repr = ((sign << EXP_WIDTH+FRAC_WIDTH)
                      + (exp << FRAC_WIDTH)
                      + cumm_frac)
    if minifloat2decimal(minifloat_repr) != decimal:
        print(f'WARNING: Check {decimal} provided. Returning '
              f'{minifloat2decimal(minifloat_repr)}')

    return minifloat_repr
decimal2minifloat = Function(decimal2minifloat, arg_units=[1], return_unit=1)


def minifloat2decimal(bitstring):
    """ Converts the 8-bit floating point representation to a decimal
        full-precision (according to python) representation.

    Args:
        bitstring (int, float, list or numpy.array): Binary word.
    Returns:
        float: Converted value in decimal.
    """
    if (isinstance(bitstring, int) or isinstance(bitstring, float)
            or isinstance(bitstring, list)):
        bitstring = np.array([bitstring])
    bitstring = bitstring.astype(int)
    val_sign, val_exponent, val_abs, val_normal = extract_fields(bitstring)
    e0 = val_exponent - val_normal - BIAS + 1
    signal = val_sign*(-2) + 1
    fraction = np.vectorize(np.binary_repr)(val_abs & 0x7, width=3)
    # Add implicit ones of normal values
    dec_val = val_normal.astype(int) * float(2)**(e0)
    # Add fractions for each element in the vector
    for idx, bits in enumerate(fraction):
        e0[idx] -= 1
        for bit in bits:
            dec_val[idx] += int(bit)*float(2)**e0[idx]
            e0[idx] -= 1
    return dec_val*signal


def fp8_multiply(num1, num2, _vectorisation_idx):
    """ Implements an 8-bit floating point multiplication scheme.

    Args:
        num1, num2 (list or numpy.array): 8-bit floating point binary word
            with MSB bits representing polarity, 3 LSB bits representing
            fraction, and intermediary bits representing exponent.
    Returns:
        list or numpy.array: 8-bit floating point binary word in the same
        format as the input.
    """
    if isinstance(_vectorisation_idx, int):
        unpack = True
        num1 = np.array([num1])
        num2 = np.array([num2])
        _vectorisation_idx = np.array([_vectorisation_idx])
    else:
        unpack = False
    num1 = num1.astype(int)
    num2 = num2.astype(int)
    val1_sign, val1_exponent, val1_abs, val1_normal = extract_fields(num1)
    val2_sign, val2_exponent, val2_abs, val2_normal = extract_fields(num2)
    result_sign = val1_sign ^ val2_sign

    # Multiplication with FRAC_WIDTH+1 LSBs. In hardware, aligning after result
    # is obtained when subnormal values are involved gives us time to calculate
    # multiplication and number of leading zeros in parallel.
    int_repr1_abs = val1_normal<<FRAC_WIDTH | (val1_abs & FRAC_MASK)
    int_repr2_abs = val2_normal<<FRAC_WIDTH | (val2_abs & FRAC_MASK)
    result = int_repr1_abs * int_repr2_abs

    # Normalization of result is done after multiplication, according
    # to leading zeros of multiplicands. Note that if multiplicand is normal,
    # no need to shift result e.g. 1.000*0.001=1000 => 1000>>2 = 100000
    # Result occupies N_BITS bits in the form of cnr1r2...r6
    num_leading_zero1 = (get_leading_zeros(int_repr1_abs)
                         - (SIGN_WIDTH + EXP_WIDTH - val1_normal))
    result = np.where(val1_normal == 0, result << num_leading_zero1, result)
    num_leading_zero2 = (get_leading_zeros(int_repr2_abs)
                         - (SIGN_WIDTH + EXP_WIDTH - val2_normal))
    result = np.where(val2_normal == 0, result << num_leading_zero2, result)
    carry = result >> (N_BITS-1)

    # This represents biased exponent - 1. Implicit normal bit is added
    # afterwards to correctly pack the result
    result_exponent = (val1_exponent - val1_normal
                       + val2_exponent - val2_normal
                       - num_leading_zero1 - num_leading_zero2
                       + (1 - BIAS) + carry)

    # In hardware, a slightly larger exponent range (herein 2 extra bits) to
    # identify underflows is needed.
    aux_ind = result_exponent >> EXP_WIDTH+1 != 0
    result[aux_ind] >>= -result_exponent[aux_ind]
    # Note that no sticky bits are computed from eliminated bits
    result_exponent[aux_ind] = EMIN

    trunc_result = np.empty_like(_vectorisation_idx)
    guard_bit = np.empty_like(_vectorisation_idx)
    sticky_bit = np.empty_like(_vectorisation_idx)

    guard_bit = result >> GUARD_WIDTH-1+carry & 1
    sticky_bit = result << N_BITS-GUARD_WIDTH+1-carry & REPR_MASK != 0

    # MSB of truncated result is included so that it is added to exponent,
    # which was calculated as the final value minus 1
    trunc_result = result >> (GUARD_WIDTH+carry)
    round_factor = guard_bit & (trunc_result|sticky_bit)
    trunc_result = trunc_result + round_factor

    result_abs = (result_exponent<<FRAC_WIDTH) + trunc_result
    # Dealing with overflow. Note that sign bit is used for comparison, so no
    # extra space would be needed in hardware
    aux_ind = result_abs > 2**(N_BITS-1) - 1
    result_abs[aux_ind] = 2**(N_BITS-1) - 1

    # Note that negative zeros are not return from operations
    result_sign[result_abs==0] = 0

    if unpack:
        return (result_sign[0] << EXP_WIDTH+FRAC_WIDTH) + result_abs[0]
    else:
        return (result_sign << EXP_WIDTH+FRAC_WIDTH) + result_abs
fp8_multiply = Function(fp8_multiply, arg_units=[1, 1],
    return_unit=1, auto_vectorise=True)


def fp8_add(num1, num2, _vectorisation_idx):
    """ Implements an 8-bit floating point addition scheme. This function was
        created to be used in an "on_pre" statement that will increment a
        post-synaptic variable. This function therefore is not suitable for
        vectorized operations because parameters involved are always scalars.

    Args:
        num1, num2 (list or numpy.array): 8-bit floating point binary word
            with MSB bits representing polarity, 3 LSB bits representing
            fraction, and intermediary bits representing exponent.
    Returns:
        list or numpy.array: 8-bit floating point binary word in the same
        format as the input.
    """
    if isinstance(_vectorisation_idx, int):
        unpack = True
        num1 = np.array([num1])
        num2 = np.array([num2])
        _vectorisation_idx = np.array([_vectorisation_idx])
    else:
        unpack = False
    num1 = num1.astype(int)
    num2 = num2.astype(int)
    num1_sign, num1_exponent, num1_abs, num1_normal = extract_fields(num1)
    num2_sign, num2_exponent, num2_abs, num2_normal = extract_fields(num2)

    # Use largest numbers as reference to simplify code
    val1_sign = np.where(num1_abs>num2_abs, num1_sign, num2_sign)
    val2_sign = np.where(num1_abs<=num2_abs, num1_sign, num2_sign)
    val1_exponent = np.where(num1_abs>num2_abs, num1_exponent, num2_exponent)
    val2_exponent = np.where(num1_abs<=num2_abs, num1_exponent, num2_exponent)
    val1_abs = np.where(num1_abs>num2_abs, num1_abs, num2_abs)
    val2_abs = np.where(num1_abs<=num2_abs, num1_abs, num2_abs)
    val1_normal = np.where(num1_abs>num2_abs, num1_normal, num2_normal)
    val2_normal = np.where(num1_abs<=num2_abs, num1_normal, num2_normal)
    result_sign = val1_sign

    opposite_signs = val1_sign ^ val2_sign
    # Note magnitude difference of normal and subnormal values, e.g. 1.0*2^-6
    # and 0.1*2^-6 have the same magnitude
    magnitude_factor = (val1_exponent - val1_normal
                        - (val2_exponent - val2_normal))

    # Get integer representation in the form of c,n,f1,f2...f6
    aux_val = ((val1_abs << (EXP_WIDTH+SIGN_WIDTH))
               & REPR_MASK)
    int_repr1_abs = ((val1_normal << (FRAC_WIDTH+GUARD_WIDTH))
                     | (aux_val >> (EXP_WIDTH+SIGN_WIDTH-GUARD_WIDTH)))
    aux_val = ((val2_abs << (EXP_WIDTH+SIGN_WIDTH))
               & REPR_MASK)
    int_repr2_abs = ((val2_normal << (FRAC_WIDTH+GUARD_WIDTH))
                     | (aux_val >> (EXP_WIDTH+SIGN_WIDTH-GUARD_WIDTH)))

    # Align smallest to largest operand's magnitude. Now smallest exponent
    # equals largest exponent
    aligned_repr2 = int_repr2_abs >> magnitude_factor

    # Uses sticky bit to avoid losing all bits when aligning
    sticky_bit = np.empty_like(_vectorisation_idx)
    # Guard bits are initialized as zero, so no bits are lost
    sticky_bit[magnitude_factor <= GUARD_WIDTH] = 0
    # All relevant bits were discarded, so worst case scenario is considered
    sticky_bit[magnitude_factor >= GUARD_WIDTH+FRAC_WIDTH] = 1
    aux_ind = ((magnitude_factor > GUARD_WIDTH)
               & (magnitude_factor < GUARD_WIDTH+FRAC_WIDTH))
    discarded_bits = (int_repr2_abs[aux_ind]
                      << (N_BITS - magnitude_factor[aux_ind]))
    sticky_bit[aux_ind] = (discarded_bits & REPR_MASK) != 0

    result = int_repr1_abs + (-1)**opposite_signs * (aligned_repr2|sticky_bit)

    num_leading_zero = get_leading_zeros(result)
    carry = result >> (N_BITS-1)
    trunc_result = np.empty_like(_vectorisation_idx)
    result_exponent = np.empty_like(_vectorisation_idx)
    guard_bit = np.empty_like(_vectorisation_idx)

    # Note that bias exponent -1 is calculated so later on result
    # containing the implicit bit is added to it.
    result_exponent = val1_exponent - val1_normal + carry

    # Subnormal result or requiring renormalization. Particularly
    # useful when dealing with underflows
    aux_ind = num_leading_zero >= 2
    num_shifts = np.where(
        val1_exponent[aux_ind]-val1_normal[aux_ind] < num_leading_zero[aux_ind],
        val1_exponent[aux_ind] - val1_normal[aux_ind],
        # Shift subtracted by -1 to align MSB to normal bit
        num_leading_zero[aux_ind] - 1)
    result_exponent[aux_ind] -= num_shifts
    result[aux_ind] <<= num_shifts

    aux_shift = GUARD_WIDTH + carry
    trunc_result = result >> aux_shift
    discarded_bits = ((result << N_BITS - aux_shift & REPR_MASK)
                      >> (N_BITS - aux_shift))
    sticky_bit = discarded_bits & (2**(aux_shift - 1) - 1) != 0
    guard_bit = discarded_bits & 2**(aux_shift - 1) != 0

    # Note that negative zeros are not return from operations
    aux_ind = np.logical_and(val1_abs==val2_abs, opposite_signs)
    result_exponent[aux_ind] = 0
    result_sign[aux_ind] = 0

    # Using LSB of trunc_result only. Guard bit together with LSB define
    # previous and next values (for round down and up, respectively)
    round_factor = guard_bit & (trunc_result | sticky_bit)
    trunc_result = trunc_result + round_factor
    result_abs = (result_exponent<<FRAC_WIDTH) + trunc_result
    # Dealing with overflow. Note that sign bit is used for comparison, so no
    # extra space would be needed in hardware
    aux_ind = result_abs > 2**(N_BITS-1) - 1
    result_abs[aux_ind] = 2**(N_BITS-1) - 1

    if unpack:
        return (result_sign[0] << EXP_WIDTH+FRAC_WIDTH) + result_abs[0]
    else:
        return (result_sign << EXP_WIDTH+FRAC_WIDTH) + result_abs
fp8_add = Function(fp8_add, arg_units=[1, 1], return_unit=1,
    auto_vectorise=True)

def fp8_smaller_than(num1, num2, _vectorisation_idx):
    if isinstance(_vectorisation_idx, int):
        unpack = True
        num1 = np.array([num1])
        num2 = np.array([num2])
        _vectorisation_idx = np.array([_vectorisation_idx])
    else:
        unpack = False
    num1 = num1.astype(int)
    num2 = num2.astype(int)
    num1_sign, num1_exponent, num1_abs, num1_normal = extract_fields(num1)
    num2_sign, num2_exponent, num2_abs, num2_normal = extract_fields(num2)

    result = np.empty_like(_vectorisation_idx)
    result[num1_sign > num2_sign] = 1
    result[num1_sign < num2_sign] = 0
    mask = num1_sign == num2_sign
    result[np.logical_and(mask, num1_abs < num2_abs)] = 1
    result[np.logical_and(mask, num1_abs >= num2_abs)] = 0

    if unpack:
        return result[0]
    else:
        return result
fp8_smaller_than = Function(fp8_smaller_than, arg_units=[1, 1],
    return_unit=1, return_type='integer', auto_vectorise=True)

DEFAULT_FUNCTIONS.update({'stochastic_decay': stochastic_decay,
                          'fp8_multiply': fp8_multiply,
                          'fp8_add': fp8_add,
                          'fp8_smaller_than': fp8_smaller_than,
                          'deterministic_decay': deterministic_decay,
                          'decimal2minifloat': decimal2minifloat})
