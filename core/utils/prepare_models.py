import numpy as np


def generate_connection_indices(pre_size, post_size, prob_conn, seed=None):
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
    Returns
    -------
    sources, targets : list of int
    """
    rng = np.random.default_rng(seed)
    conn_mat = rng.choice([0, 1],
                                 size=(pre_size, post_size),
                                 p=[1-prob_conn, prob_conn])
    sources, targets = conn_mat.nonzero()

    return sources, targets
