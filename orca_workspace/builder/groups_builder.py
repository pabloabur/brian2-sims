import sys
from warnings import warn

from brian2 import Synapses, NeuronGroup, ExplicitStateUpdater
from brian2.units import *

method = ExplicitStateUpdater('''x_new = f(x,t)''')

def create_synapses(source, target, model_desc, raise_warning=False,
                    name='synapses*'):
    syn_group = Synapses(source, target,
                         model=model_desc.model,
                         on_pre=model_desc.on_pre,
                         on_post=model_desc.on_post,
                         namespace=model_desc.namespace,
                         name=name,
                         method=method)
    syn_group.connect(**model_desc.connection)
    set_params(syn_group, model_desc.parameters, raise_warning)

    return syn_group

def create_neurons(num_neurons, model_desc, raise_warning=False, 
                   name='neurongroup*'):
    neu_group = NeuronGroup(num_neurons,
                            model=model_desc.model,
                            threshold=model_desc.threshold,
                            reset=model_desc.reset,
                            refractory=model_desc.refractory,
                            namespace=model_desc.namespace,
                            name=name,
                            method=method)
    set_params(neu_group, model_desc.parameters, raise_warning)

    return neu_group

def set_params(group, parameters, raise_warning):
    for param, val in parameters.items():
        try:
            setattr(group, param, val)
            if raise_warning:
                if 0 in getattr(group, param):
                    warn(f'Variable {param} was initialized as zero with '
                         f'set_params. Please make sure that dictionary '
                         f'provided is ordered so that a non-initialized '
                         f'variable will not erroneously set to zero other '
                         f'variables dependending on it. You can ignore '
                         f'this warning by setting raise_warning=True')
        except AttributeError as e:
            # TODO maybe this is not necessary because of dict definitions
            raise type(e)(f'{e} Group {group.name} has no state variable '
                          f'{param}').with_traceback(sys.exc_info()[2])
