import numpy as np


def generate_connection_indices(pre_size, post_size, prob_conn, seed=None,
                                allow_autapse=True):
    """ Get indices randomly manually. This is useful when we want
        to have a variable that have all the connections before-hand.

    Parameters
    ----------
    pre_size, post_size : int
        Size of pre and postsynaptic population
    prob_conn : float
        Probability of connecting to post
    seed : int
        seed of rng
    allow_autapse : boolean
        Wether connections from neurons to themselves should be removed. This
        can happen in a recurrent network i.e. source and target neuronal groups
        are the same

    Returns
    -------
    sources, targets : list of int
    """
    rng = np.random.default_rng(seed)
    conn_mat = rng.choice([0, 1],
                                 size=(pre_size, post_size),
                                 p=[1-prob_conn, prob_conn])
    sources, targets = conn_mat.nonzero()

    if not allow_autapse:
        del_items = sources == targets
        sources = [x for i, x in enumerate(sources) if not del_items[i]]
        targets = [x for i, x in enumerate(targets) if not del_items[i]]

    return sources, targets

def set_hardwarelike_scheme(prefs, neurons, run_reg_dt, precision):
    """ Required function to set a simulation scheme similar to hardware
        proposed (see Wang et al., 2018).
        
    Parameters
    ----------
    prefs : brian2.prefs
        Class containing information of preferred simulation settings
    neurons : list of brian2.NeuronGroup
        Neurons to which run_regularly operations will be added to.
    run_reg_dt : brian2.ms
        Indicates how often neurons' run_regularly is performed.
    precision: str
        Data type to be used. Currently supports 'fp8' and 'fp64',
        where each has different arithmetics to be performed.
    """
    prefs.core.network.default_schedule = ['start', 'groups', 'thresholds',
                                           'resets', 'synapses', 'end']
    if precision == 'fp64':
        ca_arith = 'Ca = Ca*int(Ca>0) - Ca*int(Ca<0)'
    elif precision == 'fp8':
        ca_arith = 'Ca = Ca*int(Ca<128) + (Ca - 128)*int(Ca>128)'
    for neu in neurons:
        neu.run_regularly(
            ca_arith,
            name=f'clear_{neu.name}_spike_flag',
            dt=run_reg_dt,
            when='after_synapses',
            order=1)
        neu.set_event_schedule('active_Ca', when='after_resets')
