import sys
from warnings import warn
import copy

from brian2 import Synapses, NeuronGroup, ExplicitStateUpdater, get_device
from brian2.units import *

from core.equations.base_equation import ParamDict

method = ExplicitStateUpdater('''x_new = f(x,t)''')

def preprocess_model_description(model_description):
    """
    At the point where groups are actually created, the constraints
    added previously are not helpful or clash with Brian2 processing. So
    this function preprocess model description to remove instances of ParamDict
    and prevent changes to affect dictionary structure.
    """
    desc = copy.copy(model_description)
    attributes = [x for x in dir(desc) if not x.startswith('_')
                                          and not callable(getattr(desc, x))]
    for attr in attributes:
        value = getattr(desc, attr)
        if issubclass(ParamDict, type(value)):
            setattr(desc, attr, dict(value))

    return desc

def create_synapses(source, target, model_desc, raise_warning=False,
                    name='synapses*'):
    proc_model_desc = preprocess_model_description(model_desc)
    syn_group = Synapses(source, target,
                         model=proc_model_desc.model,
                         on_pre=proc_model_desc.on_pre,
                         on_post=proc_model_desc.on_post,
                         on_event=proc_model_desc.on_event,
                         namespace=proc_model_desc.namespace,
                         name=name,
                         method=method)
    syn_group.connect(**proc_model_desc.connection)
    set_params(syn_group, proc_model_desc.parameters, raise_warning)

    return syn_group

def create_neurons(num_neurons, model_desc, raise_warning=False, 
                   name='neurongroup*'):
    proc_model_desc = preprocess_model_description(model_desc)
    neu_group = NeuronGroup(num_neurons,
                            model=proc_model_desc.model,
                            threshold=proc_model_desc.threshold,
                            reset=proc_model_desc.reset,
                            refractory=proc_model_desc.refractory,
                            events=proc_model_desc.events,
                            namespace=proc_model_desc.namespace,
                            name=name,
                            method=method)
    set_params(neu_group, proc_model_desc.parameters, raise_warning)

    return neu_group

def set_params(group, parameters, raise_warning):
    for param, val in parameters.items():
        setattr(group, param, val)
        if raise_warning:
            if get_device().__class__.__name__ == 'CPPStandaloneDevice':
                warn(f'Could not check warning conditions. This can be '
                     f'caused by cpp standalone mode, which does not '
                     f'allow access to variables before the network '
                     f'is run. If possible, run a quick simulation '
                     f'with numpy simulation to check warnings.')
            elif 0 in getattr(group, param):
                warn(f'Variable {param} was initialized as zero with '
                     f'set_params. Please make sure that dictionary '
                     f'provided is ordered so that a non-initialized '
                     f'variable will not erroneously set to zero other '
                     f'variables dependending on it (e.g. ..., "alpha": '
                     f' "tau_m/(dt + tau_m)", "tau": "20*ms", ... will '
                     f'result in alpha set to 0). You can ignore this '
                     f'warning by setting raise_warning=False')
