from brian2 import implementation, check_units, ms, declare_types,\
        SpikeMonitor, Network, NeuronGroup, TimedArray, Function,\
        DEFAULT_FUNCTIONS
import numpy as np
import os
from warnings import warn
from bisect import bisect_left
current_dir = os.path.abspath(os.path.dirname(__file__))

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

def stochastic_decay(init_value, decay_numerator, _vectorisation_idx):
    """ This function implements an stochastic exponential decay suited
        for digitial hardware implementations. It is mathematically
        described as V = int(V*tau/(tau + dt) + rand()) when data types
        have 4-bits.
    Args:
        init_value (list or numpy.array): Values to be decayed
        decay_numerator (int): Values that when divided by 256
            represents decay rate, that is tau/(tau + dt)
    """
    rand_num_bits = 4
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
stochastic_decay = Function(stochastic_decay, arg_units=[1, 1],
                            return_unit=1, stateless=False,
                            auto_vectorise=True)
cpp_code = '''
int stochastic_decay(int init_value, int decay_numerator, int _vectorisation_idx)
{
    int new_val;
    int init_value_sign;
    int lfsr;
    int values = 15;
    int rand_num_bits = 4;

    if (init_value < 0)
        init_value_sign = -1;
    else
        init_value_sign = 1;
    init_value = abs(init_value);

    lfsr = ceil(rand(_vectorisation_idx) * values);
    new_val = init_value<<rand_num_bits | lfsr;
    new_val = (new_val*decay_numerator) >> 8;
    new_val = new_val >> rand_num_bits;
    return init_value_sign * new_val;
}
'''
stochastic_decay.implementations.add_implementation('cpp', cpp_code,
    dependencies={'rand': DEFAULT_FUNCTIONS['rand'],
                  'abs': DEFAULT_FUNCTIONS['abs'],
                  'ceil': DEFAULT_FUNCTIONS['ceil']})

def deterministic_decay(init_value, decay_numerator):
    """ This function implements an exponential decay suited
        for digitial hardware implementations. It is mathematically
        described as V = V*tau/(tau + dt)) when data types have 8-bits.
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
                            return_unit=1)
cpp_code = '''
int deterministic_decay(int init_value, int decay_numerator)
{
    int new_val;
    int init_value_sign;

    if (init_value < 0)
        init_value_sign = -1;
    else
        init_value_sign = 1;
    init_value = abs(init_value);

    new_val = (init_value*decay_numerator) >> 8;
    return init_value_sign * new_val;
}
'''
deterministic_decay.implementations.add_implementation('cpp', cpp_code,
    dependencies={'abs': DEFAULT_FUNCTIONS['abs']})


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

def decimal2minifloat(decimal_value, bitwise_logic=False, raise_warning=True):
    """ Converts a representable decimal value to 8-bit floating point.
        Use it with CAUTION as it does not check if decimal provided is
        actually supported in the implemented format.

    Args:
        decimal_value (int, float, list or numpy.array): decimal value to be
            converted
        bitwise_logic (boolean): DEPRECATED. It is maintained here just for
            reference, although some errors are still present e.g. it fails
            to convert 0.0019. Indicates whether bitwise logic should be used
            to convert decimal
    Returns:
        (list or int): minifloat representation of decimal.
    """
    if bitwise_logic:
        if decimal_value == 0:
            return 0

        sign = 0 if decimal_value >= 0 else 1
        # TODO str includes square brackets
        # TODO try plot_fp8 as well
        decimal_abs = np.abs(decimal_value)
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
        if minifloat2decimal(minifloat_repr) != decimal_value:
            print(f'WARNING: Check {decimal_value} provided. Returning '
                  f'{minifloat2decimal(minifloat_repr)}')
    else:
        minifloats = [x for x in range(256)]
        decimals = minifloat2decimal(minifloats).tolist()
        # In python 0==-0. We want to avoid these two to be a single key below
        decimals[128] = '-0'
        minifloat_map = dict(zip(decimals, minifloats))

        if isinstance(decimal_value, int) or isinstance(decimal_value, float):
            decimal_value = np.array([decimal_value])
        else:
            decimal_value = np.array(decimal_value)

        non_matching = [x for x in decimal_value if x not in minifloat_map]

        if non_matching:
            if raise_warning:
                warn(f'WARNING: Values {non_matching} are not representable. '
                     f'Returning nearest representation. You can ignore this '
                     f'warning by setting raise_warning=False')
            aux_decimals = sorted([x for x in minifloat_map.keys() if x!='-0'])
            aux_decimals = np.array(aux_decimals)
            ideal_position = list(map(lambda x: bisect_left(aux_decimals, x),
                                      non_matching))
            ideal_position = np.clip(ideal_position, 0, len(aux_decimals) - 1)

            previous_ind = np.clip(ideal_position - 1, 0, len(aux_decimals) - 1)
            previous_values = aux_decimals[previous_ind]
            next_ind = ideal_position
            next_values = aux_decimals[next_ind]

            rounded_values = np.where(
                (np.abs(next_values-non_matching)
                 < np.abs(non_matching-previous_values)),
                next_values, previous_values)

            non_matching_id = np.array([i for i, x in enumerate(decimal_value)
                                            if x not in minifloat_map])
            decimal_value[non_matching_id] = rounded_values

    if len(decimal_value)==1:
        minifloat_representation = [minifloat_map[x] for x in decimal_value][0]
    else:
        minifloat_representation = [minifloat_map[x] for x in decimal_value]
    return minifloat_representation


def minifloat2decimal(bitstring):
    """ Converts the 8-bit floating point representation to a decimal
        full-precision (according to python) representation.

    Parameters
    ----------
    bitstring : int, float, list or numpy.array
        Binary word. In case of a floating point number, fractional part is
        discarded with int casting.

    Returns
    -------
    float
        Converted value in decimal.

    Notes
    -----
    For small numbers, it is worth setting np.set_printoptions(precision=20)
    so there is no truncation.
    """
    if isinstance(bitstring, int) or isinstance(bitstring, float):
        bitstring = np.array([bitstring])
    else:
        bitstring = np.array(bitstring)

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
    unpack = False
    if isinstance(_vectorisation_idx, int):
        unpack = True
        num1 = np.array([num1])
        num2 = np.array([num2])
        _vectorisation_idx = np.array([_vectorisation_idx])

    # Handle cases where single parameter from single object in involved with
    # arrays
    if isinstance(num1, int):
        num1 = np.array([num1 for _ in _vectorisation_idx])
    if isinstance(num2, int):
        num2 = np.array([num2 for _ in _vectorisation_idx])

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
    # no need to shift result e.g. 1.000*0.001=1000 => 1000<<3 = 1.000000
    # Result occupies N_BITS bits in the form of cnr1r2...r6
    num_leading_zero1 = get_leading_zeros(int_repr1_abs) - EXP_WIDTH
    result = np.where(val1_normal == 0, result << num_leading_zero1, result)
    num_leading_zero2 = get_leading_zeros(int_repr2_abs) - EXP_WIDTH
    result = np.where(val2_normal == 0, result << num_leading_zero2, result)
    carry = result >> (N_BITS-1)

    # This represents biased exponent - 1. Implicit normal bit is added
    # afterwards to correctly pack the result
    result_exponent = (val1_exponent - val1_normal
                       + val2_exponent - val2_normal
                       - num_leading_zero1 - num_leading_zero2
                       + (1 - BIAS) + carry)

    # In hardware, a slightly larger exponent range (herein 1 extra bit) to
    # handle subnormals
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
cpp_code = """
int fp8_multiply(int num1, int num2, int _vectorisation_idx){
    unsigned char sign1, exp1, abs_val1, sign2, exp2, abs_val2;
    unsigned char int_repr1_abs, int_repr2_abs, aux_int_repr, carry;
    unsigned char num_leading_zero1=0, num_leading_zero2=0;
    unsigned char result, result_sign, result_exp, trunc_result, result_abs;
    unsigned char guard_bit, sticky_bit, round_factor;
    bool is_normal1, is_normal2;

    const unsigned char EXP_WIDTH = 4;
    const unsigned char FRAC_WIDTH = 3;
    const unsigned char FRAC_MASK = (1<<FRAC_WIDTH) - 1;
    const unsigned char SIGN_WIDTH = 1;
    const unsigned char N_BITS = SIGN_WIDTH + EXP_WIDTH + FRAC_WIDTH;
    const unsigned char BIAS = 7;
    const unsigned char GUARD_WIDTH = 3;
    const unsigned char REPR_MASK =  (1<<N_BITS) - 1;
    // Smallest normal: 2^-(-BIAS+1)
    // Smallest subnormal: 2^(-BIAS+1)/2^(FRAC_WIDTH-1)
    // Biased representation of exponents, i.e. what is actually stored in hardware
    const char EMIN = 0;

    // Code to extract relevant fields of the bitstring
    sign1 = num1 >> (EXP_WIDTH+FRAC_WIDTH);
    abs_val1 = num1 & 0x7F;
    exp1 = abs_val1 >> FRAC_WIDTH;
    is_normal1 = abs_val1 >= (1 << FRAC_WIDTH);

    // Code to extract relevant fields of the bitstring
    sign2 = num2 >> (EXP_WIDTH+FRAC_WIDTH);
    abs_val2 = num2 & 0x7F;
    exp2 = abs_val2 >> FRAC_WIDTH;
    is_normal2 = abs_val2 >= (1 << FRAC_WIDTH);

    result_sign = sign1 ^ sign2;

    // Multiplication with FRAC_WIDTH+1 LSBs. In hardware, aligning after result
    // is obtained when subnormal values are involved gives us time to calculate
    // multiplication and number of leading zeros in parallel.
    int_repr1_abs = (is_normal1 << FRAC_WIDTH) | (abs_val1 & FRAC_MASK);
    int_repr2_abs = (is_normal2 << FRAC_WIDTH) | (abs_val2 & FRAC_MASK);
    // result occupies N_BITS bits in the form of cnr1r2...r6
    result = int_repr1_abs * int_repr2_abs;

    // Code to extract number of leading bits
    if (int_repr1_abs==0){
        num_leading_zero1 = N_BITS;
    }else{
        aux_int_repr = int_repr1_abs;
        if(aux_int_repr<=0x0F) {aux_int_repr<<=4; num_leading_zero1+=4;}
        if(aux_int_repr<=0x3F) {aux_int_repr<<=2; num_leading_zero1+=2;}
        if(aux_int_repr<=0x7F) {aux_int_repr<<=1; num_leading_zero1+=1;}
    }

    // Normalization of result is done after multiplication, according
    // to leading zeros of multiplicands. Note that if multiplicand is normal,
    // no need to shift result e.g. 1.000*0.001=1000 => 1000<<3 = 1.000000
    num_leading_zero1 = num_leading_zero1 - EXP_WIDTH;
    if (!is_normal1) result <<= num_leading_zero1;

    // Code to extract number of leading bits
    if (int_repr2_abs==0){
        num_leading_zero2 = N_BITS;
    }else{
        aux_int_repr = int_repr2_abs;
        if(aux_int_repr<=0x0F) {aux_int_repr<<=4; num_leading_zero2+=4;}
        if(aux_int_repr<=0x3F) {aux_int_repr<<=2; num_leading_zero2+=2;}
        if(aux_int_repr<=0x7F) {aux_int_repr<<=1; num_leading_zero2+=1;}
    }

    num_leading_zero2 = num_leading_zero2 - EXP_WIDTH;
    if (!is_normal2) result <<= num_leading_zero2;
    carry = result >> (N_BITS-1);

    // This represents biased exponent - 1. Implicit normal bit is added
    // afterwards to correctly pack the result
    result_exp = (exp1 - is_normal1 
                  + exp2 - is_normal2 
                  - num_leading_zero1 - num_leading_zero2
                  + (1 - BIAS) + carry);

    // In hardware, a slightly larger exponent range (herein 1 extra bit) to
    // handle subnormals
    if((result_exp >> (EXP_WIDTH + 1)) != 0){
        result >>= -result_exp;
        // Note that no sticky bits are computed from eliminated bits
        result_exp = EMIN;
    }

    guard_bit = (result >> (GUARD_WIDTH-1+carry)) & 1;
    sticky_bit = ((result << (N_BITS-GUARD_WIDTH+1-carry)) & REPR_MASK) != 0;

    // MSB of truncated result is included so that it is added to exponent,
    // which was calculated as the final value minus 1
    trunc_result = result >> (GUARD_WIDTH+carry);
    round_factor = guard_bit & (trunc_result|sticky_bit);
    trunc_result = trunc_result + round_factor;

    result_abs = (result_exp<<FRAC_WIDTH) + trunc_result;
    // Dealing with overflow. Note that sign bit is used for comparison, so no
    // extra space would be needed in hardware
    if (result_abs > (1 << (N_BITS-1)) - 1) result_abs = (1 << (N_BITS-1)) - 1;

    // Note that negative zeros are not return from operations
    if (result_abs==0) result_sign = 0;

    return (result_sign << (EXP_WIDTH+FRAC_WIDTH)) + result_abs;
}
"""
fp8_multiply.implementations.add_implementation('cpp', cpp_code)

def fp8_multiply_stochastic(num1, num2, _vectorisation_idx):
    """ Implements an 8-bit floating point multiplication scheme with
        stochastic rounding.

    Args:
        num1, num2 (list or numpy.array): 8-bit floating point binary word
            with MSB bits representing polarity, 3 LSB bits representing
            fraction, and intermediary bits representing exponent.
    Returns:
        list or numpy.array: 8-bit floating point binary word in the same
        format as the input.
    """
    unpack = False
    if isinstance(_vectorisation_idx, int):
        unpack = True
        num1 = np.array([num1])
        num2 = np.array([num2])
        _vectorisation_idx = np.array([_vectorisation_idx])

    # Handle cases where single parameter from single object in involved with
    # arrays
    if isinstance(num1, int):
        num1 = np.array([num1 for _ in _vectorisation_idx])
    if isinstance(num2, int):
        num2 = np.array([num2 for _ in _vectorisation_idx])

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
    # no need to shift result e.g. 1.000*0.001=1000 => 1000<<3 = 1.000000
    # Result occupies N_BITS bits in the form of cnr1r2...r6
    num_leading_zero1 = get_leading_zeros(int_repr1_abs) - EXP_WIDTH
    result = np.where(val1_normal == 0, result << num_leading_zero1, result)
    num_leading_zero2 = get_leading_zeros(int_repr2_abs) - EXP_WIDTH
    result = np.where(val2_normal == 0, result << num_leading_zero2, result)
    carry = result >> (N_BITS-1)

    # This represents biased exponent - 1. Implicit normal bit is added
    # afterwards to correctly pack the result
    result_exponent = (val1_exponent - val1_normal
                       + val2_exponent - val2_normal
                       - num_leading_zero1 - num_leading_zero2
                       + (1 - BIAS) + carry)

    # In hardware, a slightly larger exponent range (herein 1 extra bit) to
    # handle subnormals
    aux_ind = result_exponent >> EXP_WIDTH+1 != 0
    result[aux_ind] >>= -result_exponent[aux_ind]
    # Note that no sticky bits are computed from eliminated bits
    result_exponent[aux_ind] = EMIN

    smaller_factor = np.where(val1_abs < val2_abs, val1_abs, val2_abs)
    aux_ind = smaller_factor == 0
    result_exponent[aux_ind] = EMIN

    trunc_result = np.empty_like(_vectorisation_idx)

    discarded_bits = (result & (FRAC_MASK<<carry)) >> carry
    lfsr = np.floor(np.random.rand(len(_vectorisation_idx)) * (2**GUARD_WIDTH)).astype(int)

    # MSB of truncated result is included so that it is added to exponent,
    # which was calculated as the final value minus 1
    trunc_result = result >> (GUARD_WIDTH+carry)
    round_factor = (discarded_bits+lfsr) >> GUARD_WIDTH
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
fp8_multiply_stochastic = Function(fp8_multiply_stochastic, arg_units=[1, 1],
    return_unit=1, auto_vectorise=True)
cpp_code = """
int fp8_multiply_stochastic(int num1, int num2, int _vectorisation_idx){
    unsigned char sign1, exp1, abs_val1, sign2, exp2, abs_val2, smaller_factor;
    unsigned char int_repr1_abs, int_repr2_abs, aux_int_repr, carry;
    unsigned char num_leading_zero1=0, num_leading_zero2=0;
    unsigned char result, result_sign, result_exp, trunc_result, result_abs;
    unsigned char discarded_bits, lfsr, round_factor;
    bool is_normal1, is_normal2;

    const unsigned char EXP_WIDTH = 4;
    const unsigned char FRAC_WIDTH = 3;
    const unsigned char FRAC_MASK = (1<<FRAC_WIDTH) - 1;
    const unsigned char SIGN_WIDTH = 1;
    const unsigned char N_BITS = SIGN_WIDTH + EXP_WIDTH + FRAC_WIDTH;
    const unsigned char BIAS = 7;
    const unsigned char GUARD_WIDTH = 3;
    const unsigned char REPR_MASK =  (1<<N_BITS) - 1;
    // Smallest normal: 2^-(-BIAS+1)
    // Smallest subnormal: 2^(-BIAS+1)/2^(FRAC_WIDTH-1)
    // Biased representation of exponents, i.e. what is actually stored in hardware
    const char EMIN = 0;

    // Code to extract relevant fields of the bitstring
    sign1 = num1 >> (EXP_WIDTH+FRAC_WIDTH);
    abs_val1 = num1 & 0x7F;
    exp1 = abs_val1 >> FRAC_WIDTH;
    is_normal1 = abs_val1 >= (1 << FRAC_WIDTH);

    // Code to extract relevant fields of the bitstring
    sign2 = num2 >> (EXP_WIDTH+FRAC_WIDTH);
    abs_val2 = num2 & 0x7F;
    exp2 = abs_val2 >> FRAC_WIDTH;
    is_normal2 = abs_val2 >= (1 << FRAC_WIDTH);

    result_sign = sign1 ^ sign2;

    // Multiplication with FRAC_WIDTH+1 LSBs. In hardware, aligning after result
    // is obtained when subnormal values are involved gives us time to calculate
    // multiplication and number of leading zeros in parallel.
    int_repr1_abs = (is_normal1 << FRAC_WIDTH) | (abs_val1 & FRAC_MASK);
    int_repr2_abs = (is_normal2 << FRAC_WIDTH) | (abs_val2 & FRAC_MASK);
    // result occupies N_BITS bits in the form of cnr1r2...r6
    result = int_repr1_abs * int_repr2_abs;

    // Code to extract number of leading bits
    if (int_repr1_abs==0){
        num_leading_zero1 = N_BITS;
    }else{
        aux_int_repr = int_repr1_abs;
        if(aux_int_repr<=0x0F) {aux_int_repr<<=4; num_leading_zero1+=4;}
        if(aux_int_repr<=0x3F) {aux_int_repr<<=2; num_leading_zero1+=2;}
        if(aux_int_repr<=0x7F) {aux_int_repr<<=1; num_leading_zero1+=1;}
    }

    // Normalization of result is done after multiplication, according
    // to leading zeros of multiplicands. Note that if multiplicand is normal,
    // no need to shift result e.g. 1.000*0.001=1000 => 1000<<3 = 1.000000
    num_leading_zero1 = num_leading_zero1 - EXP_WIDTH;
    if (!is_normal1) result <<= num_leading_zero1;

    // Code to extract number of leading bits
    if (int_repr2_abs==0){
        num_leading_zero2 = N_BITS;
    }else{
        aux_int_repr = int_repr2_abs;
        if(aux_int_repr<=0x0F) {aux_int_repr<<=4; num_leading_zero2+=4;}
        if(aux_int_repr<=0x3F) {aux_int_repr<<=2; num_leading_zero2+=2;}
        if(aux_int_repr<=0x7F) {aux_int_repr<<=1; num_leading_zero2+=1;}
    }

    num_leading_zero2 = num_leading_zero2 - EXP_WIDTH;
    if (!is_normal2) result <<= num_leading_zero2;
    carry = result >> (N_BITS-1);

    // This represents biased exponent - 1. Implicit normal bit is added
    // afterwards to correctly pack the result
    result_exp = (exp1 - is_normal1 
                  + exp2 - is_normal2 
                  - num_leading_zero1 - num_leading_zero2
                  + (1 - BIAS) + carry);

    // In hardware, a slightly larger exponent range (herein 1 extra bit) to
    // handle subnormals
    if((result_exp >> (EXP_WIDTH + 1)) != 0){
        result >>= -result_exp;
        // Note that no sticky bits are computed from eliminated bits
        result_exp = EMIN;
    }

    smaller_factor = (abs_val1 < abs_val2) ? abs_val1 : abs_val2;
    if (smaller_factor == 0) result_exp = EMIN;

    discarded_bits = (result & FRAC_MASK<<carry) >> carry;
    lfsr = floor(rand(_vectorisation_idx) * (1 << GUARD_WIDTH));

    // MSB of truncated result is included so that it is added to exponent,
    // which was calculated as the final value minus 1
    trunc_result = result >> (GUARD_WIDTH+carry);
    round_factor = (discarded_bits+lfsr) >> GUARD_WIDTH;
    trunc_result = trunc_result + round_factor;

    result_abs = (result_exp<<FRAC_WIDTH) + trunc_result;
    // Dealing with overflow. Note that sign bit is used for comparison, so no
    // extra space would be needed in hardware
    if (result_abs > (1 << (N_BITS-1)) - 1) result_abs = (1 << (N_BITS-1)) - 1;

    // Note that negative zeros are not return from operations
    if (result_abs==0) result_sign = 0;

    return (result_sign << (EXP_WIDTH+FRAC_WIDTH)) + result_abs;
}
"""
fp8_multiply_stochastic.implementations.add_implementation('cpp', cpp_code,
    dependencies={'rand': DEFAULT_FUNCTIONS['rand'],
                  'floor': DEFAULT_FUNCTIONS['floor']})


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
    unpack = False
    if isinstance(_vectorisation_idx, int):
        unpack = True
        num1 = np.array([num1])
        num2 = np.array([num2])
        _vectorisation_idx = np.array([_vectorisation_idx])

    # Handle cases where single parameter from single object in involved with
    # arrays
    if isinstance(num1, int):
        num1 = np.array([num1 for _ in _vectorisation_idx])
    if isinstance(num2, int):
        num2 = np.array([num2 for _ in _vectorisation_idx])

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
cpp_code = """
int fp8_add(int num1, int num2, int _vectorisation_idx){
    unsigned char num1_sign, num1_exp, num1_abs, num1_normal;
    unsigned char num2_sign, num2_exp, num2_abs, num2_normal;
    unsigned char sign1, exp1, abs_val1, is_normal1;
    unsigned char sign2, exp2, abs_val2, is_normal2;
    unsigned char magnitude_factor, aux_val;
    unsigned char int_repr1_abs, int_repr2_abs, aligned_repr2, aux_int_repr;
    unsigned char sticky_bit, guard_bit, discarded_bits, round_factor;
    unsigned char result, result_sign, result_exp, trunc_result, result_abs;
    unsigned char carry, num_leading_zero=0, num_shifts, aux_shift;
    bool opposite_signs;

    const unsigned char EXP_WIDTH = 4;
    const unsigned char FRAC_WIDTH = 3;
    const unsigned char SIGN_WIDTH = 1;
    const unsigned char N_BITS = SIGN_WIDTH + EXP_WIDTH + FRAC_WIDTH;
    const unsigned char GUARD_WIDTH = 3;
    const unsigned char REPR_MASK =  (1<<N_BITS) - 1;
    // Smallest normal: 2^-(-BIAS+1)
    // Smallest subnormal: 2^(-BIAS+1)/2^(FRAC_WIDTH-1)
    // Biased representation of exponents, i.e. what is actually stored in hardware

    // Code to extract relevant fields of the bitstring
    num1_sign = num1 >> (EXP_WIDTH+FRAC_WIDTH);
    num1_abs = num1 & 0x7F;
    num1_exp = num1_abs >> FRAC_WIDTH;
    num1_normal = num1_abs >= (1 << FRAC_WIDTH);

    // Code to extract relevant fields of the bitstring
    num2_sign = num2 >> (EXP_WIDTH+FRAC_WIDTH);
    num2_abs = num2 & 0x7F;
    num2_exp = num2_abs >> FRAC_WIDTH;
    num2_normal = num2_abs >= (1 << FRAC_WIDTH);

    // Use largest numbers as reference to simplify code
    if (num1_abs>num2_abs){
        sign1 = num1_sign; sign2 = num2_sign;
        exp1 = num1_exp; exp2 = num2_exp;
        abs_val1 = num1_abs; abs_val2 = num2_abs;
        is_normal1 = num1_normal; is_normal2 = num2_normal;
        result_sign = num1_sign;
    }else{
        sign1 = num2_sign; sign2 = num1_sign;
        exp1 = num2_exp; exp2 = num1_exp;
        abs_val1 = num2_abs; abs_val2 = num1_abs;
        is_normal1 = num2_normal; is_normal2 = num1_normal;
        result_sign = num2_sign;
    }

    opposite_signs = sign1 ^ sign2;
    // Note magnitude difference of normal and subnormal values, e.g. 1.0*2^-6
    // and 0.1*2^-6 have the same magnitude
    magnitude_factor = (exp1 - is_normal1
                        - (exp2 - is_normal2));

    // Get integer representation in the form of c,n,f1,f2...f6
    aux_val = ((abs_val1 << (EXP_WIDTH+SIGN_WIDTH))
               & REPR_MASK);
    int_repr1_abs = ((is_normal1 << (FRAC_WIDTH+GUARD_WIDTH))
                     | (aux_val >> (EXP_WIDTH+SIGN_WIDTH-GUARD_WIDTH)));
    aux_val = ((abs_val2 << (EXP_WIDTH+SIGN_WIDTH))
               & REPR_MASK);
    int_repr2_abs = ((is_normal2 << (FRAC_WIDTH+GUARD_WIDTH))
                     | (aux_val >> (EXP_WIDTH+SIGN_WIDTH-GUARD_WIDTH)));

    // Align smallest to largest operand's magnitude. Now smallest exponent
    // equals largest exponent
    aligned_repr2 = int_repr2_abs >> magnitude_factor;

    // Uses sticky bit to avoid losing all bits when aligning.
    // Guard bits are initialized as zero, so no bits are lost
    if (magnitude_factor <= GUARD_WIDTH){
        sticky_bit = 0;
    } else if (magnitude_factor >= GUARD_WIDTH+FRAC_WIDTH){
        // All relevant bits were discarded, so worst case scenario is considered
        sticky_bit = 1;
    } else if ((magnitude_factor > GUARD_WIDTH)
               && (magnitude_factor < GUARD_WIDTH+FRAC_WIDTH)){
        discarded_bits = int_repr2_abs << (N_BITS - magnitude_factor);
        sticky_bit = (discarded_bits & REPR_MASK) != 0;
    }
    
    if (opposite_signs)
        result = int_repr1_abs - (aligned_repr2|sticky_bit);
    else
        result = int_repr1_abs + (aligned_repr2|sticky_bit);

    // Code to extract number of leading bits
    if (result==0){
        num_leading_zero = N_BITS;
    }else{
        aux_int_repr = result;
        if(aux_int_repr<=0x0F) {aux_int_repr<<=4; num_leading_zero+=4;}
        if(aux_int_repr<=0x3F) {aux_int_repr<<=2; num_leading_zero+=2;}
        if(aux_int_repr<=0x7F) {aux_int_repr<<=1; num_leading_zero+=1;}
    }
    carry = result >> (N_BITS-1);

    // Note that bias exponent -1 is calculated so later on result
    // containing the implicit bit is added to it.
    result_exp = exp1 - is_normal1 + carry;

    // Subnormal result or requiring renormalization. Particularly
    // useful when dealing with underflows
    if (num_leading_zero >= 2){
        if (exp1-is_normal1 < num_leading_zero)
            num_shifts = exp1 - is_normal1;
        else
            // Shift subtracted by -1 to align MSB to normal bit
            num_shifts = num_leading_zero - 1;
        result_exp -= num_shifts;
        result <<= num_shifts;
    }

    aux_shift = GUARD_WIDTH + carry;
    trunc_result = result >> aux_shift;
    discarded_bits = (((result << (N_BITS-aux_shift)) & REPR_MASK)
                      >> (N_BITS - aux_shift));
    sticky_bit = (discarded_bits & ((1 << (aux_shift-1)) - 1)) != 0;
    guard_bit = (discarded_bits & (1 << (aux_shift-1))) != 0;

    // Note that negative zeros are not return from operations
    if (abs_val1==abs_val2 && opposite_signs){
        result_exp = 0;
        result_sign = 0;
    }

    // Using LSB of trunc_result only. Guard bit together with LSB define
    // previous and next values (for round down and up, respectively)
    round_factor = guard_bit & (trunc_result | sticky_bit);
    trunc_result = trunc_result + round_factor;
    result_abs = (result_exp<<FRAC_WIDTH) + trunc_result;
    // Dealing with overflow. Note that sign bit is used for comparison, so no
    // extra space would be needed in hardware
    if (result_abs > (1<<(N_BITS-1)) - 1)
        result_abs = (1<<(N_BITS-1)) - 1;

    return (result_sign << (EXP_WIDTH+FRAC_WIDTH)) + result_abs;
}
"""
fp8_add.implementations.add_implementation('cpp', cpp_code)


def fp8_add_stochastic(num1, num2, _vectorisation_idx):
    """ Implements an 8-bit floating point addition scheme with stochastic
        rounding. This function was created to be used in an "on_pre"
        statement that will increment a post-synaptic variable. This
        function therefore is not suitable for vectorized operations
        because parameters involved are always scalars.

    Args:
        num1, num2 (list or numpy.array): 8-bit floating point binary word
            with MSB bits representing polarity, 3 LSB bits representing
            fraction, and intermediary bits representing exponent.
    Returns:
        list or numpy.array: 8-bit floating point binary word in the same
        format as the input.
    """
    unpack = False
    if isinstance(_vectorisation_idx, int):
        unpack = True
        num1 = np.array([num1])
        num2 = np.array([num2])
        _vectorisation_idx = np.array([_vectorisation_idx])

    # Handle cases where single parameter from single object in involved with
    # arrays
    if isinstance(num1, int):
        num1 = np.array([num1 for _ in _vectorisation_idx])
    if isinstance(num2, int):
        num2 = np.array([num2 for _ in _vectorisation_idx])

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

    # Uses stochastic bit to avoid losing all bits when aligning
    num_leading_zero = get_leading_zeros(int_repr2_abs)
    lower_factor = 7 - num_leading_zero
    aux_ind = magnitude_factor > lower_factor
    aux_ind[int_repr2_abs == 0] = False
    sticky_bit = np.zeros_like(_vectorisation_idx)
    low_prob_len = (magnitude_factor[aux_ind] - 7 + num_leading_zero[aux_ind])
    low_prob = np.floor(np.random.rand(len(low_prob_len)) * (2**low_prob_len)).astype(int)
    sticky_bit[aux_ind] = low_prob==(2**low_prob_len-1)

    result = int_repr1_abs + (-1)**opposite_signs * (aligned_repr2|sticky_bit)

    num_leading_zero = get_leading_zeros(result)
    carry = result >> (N_BITS-1)
    trunc_result = np.empty_like(_vectorisation_idx)
    result_exponent = np.empty_like(_vectorisation_idx)

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

    discarded_bits = np.zeros_like(_vectorisation_idx)
    aux_shift = GUARD_WIDTH + carry
    trunc_result = result >> aux_shift
    discarded_bits = (result & (FRAC_MASK<<carry)) >> carry

    # Note that negative zeros are not return from operations
    aux_ind = np.logical_and(val1_abs==val2_abs, opposite_signs)
    result_exponent[aux_ind] = 0
    result_sign[aux_ind] = 0

    lfsr_len = np.empty_like(_vectorisation_idx)
    lfsr_len[:] = GUARD_WIDTH
    lfsr = np.floor(np.random.rand(len(_vectorisation_idx)) * (2**lfsr_len)).astype(int)
    round_factor = (discarded_bits+lfsr) >> GUARD_WIDTH
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
fp8_add_stochastic = Function(fp8_add_stochastic, arg_units=[1, 1],
    return_unit=1, auto_vectorise=True)
cpp_code = """
int fp8_add_stochastic(int num1, int num2, int _vectorisation_idx){
    unsigned char num1_sign, num1_exp, num1_abs, num1_normal;
    unsigned char num2_sign, num2_exp, num2_abs, num2_normal;
    unsigned char sign1, exp1, abs_val1, is_normal1;
    unsigned char sign2, exp2, abs_val2, is_normal2;
    unsigned char magnitude_factor, aux_val;
    unsigned char int_repr1_abs, int_repr2_abs, aligned_repr2, aux_int_repr;
    unsigned char sticky_bit, low_prob_len, low_prob, lfsr, lower_factor, discarded_bits, round_factor;
    unsigned char result, result_sign, result_exp, trunc_result, result_abs;
    unsigned char carry, num_leading_zero=0, num_shifts, aux_shift;
    bool opposite_signs;

    const unsigned char EXP_WIDTH = 4;
    const unsigned char FRAC_WIDTH = 3;
    const unsigned char FRAC_MASK = (1<<FRAC_WIDTH) - 1;
    const unsigned char SIGN_WIDTH = 1;
    const unsigned char N_BITS = SIGN_WIDTH + EXP_WIDTH + FRAC_WIDTH;
    const unsigned char GUARD_WIDTH = 3;
    const unsigned char REPR_MASK =  (1<<N_BITS) - 1;
    // Smallest normal: 2^-(-BIAS+1)
    // Smallest subnormal: 2^(-BIAS+1)/2^(FRAC_WIDTH-1)
    // Biased representation of exponents, i.e. what is actually stored in hardware

    // Code to extract relevant fields of the bitstring
    num1_sign = num1 >> (EXP_WIDTH+FRAC_WIDTH);
    num1_abs = num1 & 0x7F;
    num1_exp = num1_abs >> FRAC_WIDTH;
    num1_normal = num1_abs >= (1 << FRAC_WIDTH);

    // Code to extract relevant fields of the bitstring
    num2_sign = num2 >> (EXP_WIDTH+FRAC_WIDTH);
    num2_abs = num2 & 0x7F;
    num2_exp = num2_abs >> FRAC_WIDTH;
    num2_normal = num2_abs >= (1 << FRAC_WIDTH);

    // Use largest numbers as reference to simplify code
    if (num1_abs>num2_abs){
        sign1 = num1_sign; sign2 = num2_sign;
        exp1 = num1_exp; exp2 = num2_exp;
        abs_val1 = num1_abs; abs_val2 = num2_abs;
        is_normal1 = num1_normal; is_normal2 = num2_normal;
        result_sign = num1_sign;
    }else{
        sign1 = num2_sign; sign2 = num1_sign;
        exp1 = num2_exp; exp2 = num1_exp;
        abs_val1 = num2_abs; abs_val2 = num1_abs;
        is_normal1 = num2_normal; is_normal2 = num1_normal;
        result_sign = num2_sign;
    }

    opposite_signs = sign1 ^ sign2;
    // Note magnitude difference of normal and subnormal values, e.g. 1.0*2^-6
    // and 0.1*2^-6 have the same magnitude
    magnitude_factor = (exp1 - is_normal1
                        - (exp2 - is_normal2));

    // Get integer representation in the form of c,n,f1,f2...f6
    aux_val = ((abs_val1 << (EXP_WIDTH+SIGN_WIDTH))
               & REPR_MASK);
    int_repr1_abs = ((is_normal1 << (FRAC_WIDTH+GUARD_WIDTH))
                     | (aux_val >> (EXP_WIDTH+SIGN_WIDTH-GUARD_WIDTH)));
    aux_val = ((abs_val2 << (EXP_WIDTH+SIGN_WIDTH))
               & REPR_MASK);
    int_repr2_abs = ((is_normal2 << (FRAC_WIDTH+GUARD_WIDTH))
                     | (aux_val >> (EXP_WIDTH+SIGN_WIDTH-GUARD_WIDTH)));

    // Align smallest to largest operand's magnitude. Now smallest exponent
    // equals largest exponent
    aligned_repr2 = int_repr2_abs >> magnitude_factor;

    // Uses stochastic bit to avoid losing all bits when aligning.
    // Code to extract number of leading bits
    if (int_repr2_abs==0){
        num_leading_zero = N_BITS;
    }else{
        aux_int_repr = int_repr2_abs;
        if(aux_int_repr<=0x0F) {aux_int_repr<<=4; num_leading_zero+=4;}
        if(aux_int_repr<=0x3F) {aux_int_repr<<=2; num_leading_zero+=2;}
        if(aux_int_repr<=0x7F) {aux_int_repr<<=1; num_leading_zero+=1;}
    }
    lower_factor = 7 - num_leading_zero;
    if (magnitude_factor>lower_factor && int_repr2_abs!=0){
        low_prob_len = magnitude_factor - 7 + num_leading_zero;
        low_prob = floor(rand(_vectorisation_idx) * (1 << low_prob_len));
        sticky_bit = low_prob == (1 << low_prob_len) - 1;
    } else {
        sticky_bit = 0;
    }
    
    if (opposite_signs)
        result = int_repr1_abs - (aligned_repr2|sticky_bit);
    else
        result = int_repr1_abs + (aligned_repr2|sticky_bit);

    // Code to extract number of leading bits
    num_leading_zero = 0;
    if (result==0){
        num_leading_zero = N_BITS;
    }else{
        aux_int_repr = result;
        if(aux_int_repr<=0x0F) {aux_int_repr<<=4; num_leading_zero+=4;}
        if(aux_int_repr<=0x3F) {aux_int_repr<<=2; num_leading_zero+=2;}
        if(aux_int_repr<=0x7F) {aux_int_repr<<=1; num_leading_zero+=1;}
    }
    carry = result >> (N_BITS-1);

    // Note that bias exponent -1 is calculated so later on result
    // containing the implicit bit is added to it.
    result_exp = exp1 - is_normal1 + carry;

    // Subnormal result or requiring renormalization. Particularly
    // useful when dealing with underflows
    if (num_leading_zero >= 2){
        if (exp1-is_normal1 < num_leading_zero)
            num_shifts = exp1 - is_normal1;
        else
            // Shift subtracted by -1 to align MSB to normal bit
            num_shifts = num_leading_zero - 1;
        result_exp -= num_shifts;
        result <<= num_shifts;
    }

    aux_shift = GUARD_WIDTH + carry;
    trunc_result = result >> aux_shift;
    discarded_bits = (result & FRAC_MASK<<carry) >> carry;

    // Note that negative zeros are not return from operations
    if (abs_val1==abs_val2 && opposite_signs){
        result_exp = 0;
        result_sign = 0;
    }

    lfsr = floor(rand(_vectorisation_idx) * (1 << GUARD_WIDTH));
    round_factor = (discarded_bits+lfsr) >> GUARD_WIDTH;

    trunc_result = trunc_result + round_factor;
    result_abs = (result_exp<<FRAC_WIDTH) + trunc_result;
    // Dealing with overflow. Note that sign bit is used for comparison, so no
    // extra space would be needed in hardware
    if (result_abs > (1<<(N_BITS-1)) - 1)
        result_abs = (1<<(N_BITS-1)) - 1;

    return (result_sign << (EXP_WIDTH+FRAC_WIDTH)) + result_abs;
}
"""
fp8_add_stochastic.implementations.add_implementation('cpp', cpp_code,
    dependencies={'rand': DEFAULT_FUNCTIONS['rand'],
                  'floor': DEFAULT_FUNCTIONS['floor']})

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
cpp_code = """
bool fp8_smaller_than(int num1, int num2, int _vectorisation_idx){
    unsigned char sign1, exp1, abs_val1, sign2, exp2, abs_val2;
    bool is_normal1, is_normal2, result;

    const unsigned char EXP_WIDTH = 4;
    const unsigned char FRAC_WIDTH = 3;

    // Code to extract relevant fields of the bitstring
    sign1 = num1 >> (EXP_WIDTH+FRAC_WIDTH);
    abs_val1 = num1 & 0x7F;
    exp1 = abs_val1 >> FRAC_WIDTH;
    is_normal1 = abs_val1 >= (1 << FRAC_WIDTH);

    // Code to extract relevant fields of the bitstring
    sign2 = num2 >> (EXP_WIDTH+FRAC_WIDTH);
    abs_val2 = num2 & 0x7F;
    exp2 = abs_val2 >> FRAC_WIDTH;
    is_normal2 = abs_val2 >= (1 << FRAC_WIDTH);

    if (sign1 > sign2)
        result = true;
    else if (sign1 < sign2)
        result = false;
    else if (sign1 == sign2){
        if (abs_val1 < abs_val2)
            result = true;
        else
            result = false;
    }

    return result;
}
"""
fp8_smaller_than.implementations.add_implementation('cpp', cpp_code)
