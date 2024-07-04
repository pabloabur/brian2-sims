from core.utils.misc import minifloat2decimal, decimal2minifloat,\
    fp8_add_stochastic, fp8_add, fp8_multiply, get_leading_zeros
import numpy as np
import pandas as pd
import plotext as plt
import json

from core.builder.groups_builder import create_synapses, create_neurons
from core.equations.base_equation import ParamDict
from core.equations.neurons.sfp8LIF import sfp8LIF
from core.equations.synapses.sfp8CUBA import sfp8CUBA

from brian2 import run, device, defaultclock, ms, SpikeGeneratorGroup,\
    StateMonitor

def get_likely_numbers(result, num1, num2, operation):
    """ Calculate the most and less likely numbers for a given operation
        result in a minifloat format.

    Parameters
    ----------
    result: nd.array
        The result of an arithmetic operation in minifloat representation.
    num1: nd.array
        The first operand in minifloat representation.
    num2: nd.array
        The second operand in minifloat representation.
    operation: str
        The type of arithmetic operation ('mul' for multiplication, 'add' for addition).

    Returns
    -------
    tuple
        A tuple containing three elements:
        - most_likely (nd.array): The most likely number (decimal representation)
            that is close to the ideal result.
        - less_likely (nd.array): The less likely number (decimal representation)
            that is farther from the ideal result.
        - ideal_prob_round_closer (nd.array): The probability that the ideal
            result is closer to the most likely number.
    """
    res_rounded = minifloat2decimal(result)
    res_next = minifloat2decimal(result + 1)
    aux_ind = result == 127
    res_next[aux_ind] = np.inf
    aux_ind = result == 255
    res_next[aux_ind] = -np.inf

    aux_ind = result == 0
    result[aux_ind] = 130
    res_prev = minifloat2decimal(result - 1)
    values = np.stack((res_prev, res_rounded, res_next), axis=1)

    a_dec = minifloat2decimal(num1)
    b_dec = minifloat2decimal(num2)

    if operation == 'mul':
        res_ideal = a_dec*b_dec
    if operation == 'add':
        res_ideal = a_dec + b_dec

    values = np.sort(values, axis=1)
    distance_to_lower = np.empty_like(res_ideal)
    distance_to_higher = np.empty_like(res_ideal)
    ideal_prob_round_closer = np.empty_like(res_ideal)

    lower = np.empty_like(res_ideal)
    higher = np.empty_like(res_ideal)
    aux_ind = res_ideal <= res_rounded
    lower[aux_ind] = values[aux_ind, 0]
    higher[aux_ind] = values[aux_ind, 1]
    aux_ind = res_ideal > res_rounded
    lower[aux_ind] = values[aux_ind, 1]
    higher[aux_ind] = values[aux_ind, 2]

    distance_to_lower = abs(res_ideal - lower)
    distance_to_higher = abs(higher - res_ideal)
    total_distance = distance_to_lower + distance_to_higher
    # inf is used to identify values beyond the bounds of format, in which
    # case lower and higher are the same
    aux_ind = total_distance == np.inf
    ideal_prob_round_closer[aux_ind] = 1
    aux_ind = total_distance != np.inf
    ideal_prob_round_closer[aux_ind] = (np.maximum(distance_to_lower[aux_ind],
                                                   distance_to_higher[aux_ind])
                                        / total_distance[aux_ind])
    aux_ind = res_rounded == res_ideal
    ideal_prob_round_closer[aux_ind] = 1

    # Arbitrarily takes first condition as default if distances are equal
    most_likely = np.empty_like(res_ideal)
    less_likely = np.empty_like(res_ideal)
    aux_ind = distance_to_lower <= distance_to_higher
    most_likely[aux_ind] = res_ideal[aux_ind] - distance_to_lower[aux_ind]
    less_likely[aux_ind] = res_ideal[aux_ind] + distance_to_higher[aux_ind]
    aux_ind = distance_to_higher < distance_to_lower
    most_likely[aux_ind] = res_ideal[aux_ind] + distance_to_higher[aux_ind]
    less_likely[aux_ind] = res_ideal[aux_ind] - distance_to_lower[aux_ind]

    return most_likely, less_likely, ideal_prob_round_closer


def minifloat_operations(args):
    defaultclock.dt = args.timestep * ms
    rng = np.random.default_rng()
    run_namespace = {}

    """ ================ models and helper functions ================ """
    if args.protocol == 1:
        factors = [x for x in range(256)]
        factors_x = []
        factors_y = []
        for x in factors:
            for y in range(x, 256):
                factors_x.append(x)
                factors_y.append(y)

        # Each neuron represents one test: result <- factor_x * factor_y
        n_tests = len(factors_x)
        model = sfp8LIF()
        model.model += 'factor_x : 1\nfactor_y : 1\n'
        model.model += 'op_result = fp8_multiply_stochastic(factor_x, factor_y) : 1\n'
        model.parameters = ParamDict({**model.parameters,
                                    **{'factor_x': 0,
                                       'factor_y': 0}})
        model.modify_model('parameters', factors_x, key='factor_x')
        model.modify_model('parameters', factors_y, key='factor_y')
        variables = ['factor_x', 'factor_y', 'op_result']
        operation = 'mul'
    elif args.protocol == 2:
        addends = [x for x in range(256)]
        addends_x = []
        addends_y = []
        for x in addends:
            for y in range(x, 256):
                addends_x.append(x)
                addends_y.append(y)

        # Each neuron represents one test: result <- addend_x + addend_y
        n_tests = len(addends_x)
        model = sfp8LIF()
        model.model += 'addend_x : 1\naddend_y : 1\n'
        model.model += 'op_result = fp8_add_stochastic(addend_x, addend_y) : 1\n'
        model.parameters = ParamDict({**model.parameters,
                                    **{'addend_x': 0,
                                       'addend_y': 0}})
        model.modify_model('parameters', addends_x, key='addend_x')
        model.modify_model('parameters', addends_y, key='addend_y')
        variables = ['addend_x', 'addend_y', 'op_result']
        operation = 'add'

    neu = create_neurons(n_tests, model, raise_warning=True)

    """ ================ Setting up monitors ================ """
    statemon = StateMonitor(neu,
                            variables=variables,
                            record=True)

    # Each trial is a timestep
    trials = 10000
    run(trials*ms, report='stdout', namespace=run_namespace)

    if args.backend == 'cpp_standalone' or args.backend == 'cuda_standalone':
        device.build(args.code_path)

    # First trial was taken just as a reference
    if args.protocol == 1:
        val1 = statemon.factor_x
        val2 = statemon.factor_y
    elif args.protocol == 2:
        val1 = statemon.addend_x
        val2 = statemon.addend_y
    most_likely, less_likely, ideal_prob_round_closer = get_likely_numbers(
        statemon.op_result[:, 0],
        val1[:, 0],
        val2[:, 0],
        operation)

    temp_val1 = val1[:, 0].astype(int) & 0x7F
    temp_val2 = val2[:, 0].astype(int) & 0x7F
    larger_abs = np.where((temp_val1) > temp_val2,
                          temp_val1,
                          temp_val2)
    larger_exp = larger_abs >> 3
    larger_is_normal = larger_abs >= 8
    larger_is_normal = larger_is_normal.astype(int)
    smaller_abs = np.where(temp_val1 <= temp_val2,
                           temp_val1,
                           temp_val2)
    smaller_exp = smaller_abs >> 3
    smaller_is_normal = smaller_abs >= 8
    smaller_is_normal = smaller_is_normal.astype(int)

    delta = (larger_exp - larger_is_normal
             - smaller_exp + smaller_is_normal)
    smaller_int_repr = ((smaller_abs.astype(int)&0x7) << 3) | (smaller_is_normal<<6)
    lambda_n_zeros = get_leading_zeros(smaller_int_repr)

    aux_ind = ideal_prob_round_closer == 1
    temp_array=statemon.op_result[aux_ind, :]
    unique_values = [np.unique(temp_array[x, :])
                     for x in range(np.shape(temp_array)[0])]
    for i, x in enumerate(unique_values):
        if len(x)>1:
            raise Exception(f'Unexpected result for exact computation.')

    likely_count = np.sum(statemon.op_result == np.array(decimal2minifloat(most_likely))[:, np.newaxis], axis=1)
    unlikely_count = np.sum(statemon.op_result != np.array(decimal2minifloat(most_likely))[:, np.newaxis], axis=1)

    """ =================== Saving data =================== """
    outcomes = pd.DataFrame(
        {'val_x': minifloat2decimal(val1[:, 0]),
         'val_y': minifloat2decimal(val2[:, 0]),
         'p_round_closer_ideal': ideal_prob_round_closer,
         'p_round_closer_sim': np.array(likely_count)/trials,
         'lambda': lambda_n_zeros,
         'delta': delta,
         'L': delta - 7 + lambda_n_zeros,
         'most_likely': most_likely,
         'less_likely': less_likely,
         'likely_count': likely_count,
         'unlikely_count': unlikely_count
         })
    outcomes.to_csv(f'{args.save_path}/probs.csv', index=False)

    metadata = {'protocol': args.protocol,
                'sim_type': 'standard'
                }
    with open(f'{args.save_path}/metadata.json', 'w') as f:
        json.dump(metadata, f)

    if not args.quiet:
        plt.clear_figure()
        plt.theme('clear')
        plt.plot_size(height=5, width=50)
        plt.scatter([less_likely[10000], most_likely[10000]], [0 for _ in range(2)])
        plt.scatter(minifloat2decimal(statemon.op_result[10000][0]), [0], color='red+')
        plt.show()
        print('---------------------------------------')
