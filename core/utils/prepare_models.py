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
