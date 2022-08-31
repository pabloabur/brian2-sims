"""
Created on Mon May 03 2021

@author=pabloabur

This file contains main parameters and definitions of the building block
related to quantized stochastic models described by Wang et al. (2018).
The dictionaries provided below represent motifs for connections and groups.
The descriptor will filter values according to layer and plasticity rules, so
that scaling up is more modular and (hopefully) more organized. Note that we
chose to be explicit, so there are a lot of variables. The final descriptor
however contains only specific parameters from the motifs.
"""
import os
import sys
import copy
import git

from brian2 import ms, mV, mA
import numpy as np

#from teili.models.neuron_models import QuantStochLIF as static_neuron_model
#from teili.models.synapse_models import QuantStochSyn as static_synapse_model
#from teili.models.synapse_models import QuantStochSynStdp as stdp_synapse_model
#from teili.models.builder.synapse_equation_builder import SynapseEquationBuilder
#from teili.models.builder.neuron_equation_builder import NeuronEquationBuilder

# Dictionaries used as Lookup table to construct descriptor
syn_input_prob = {
    'L4': {'ff_pyr': 0.7,
           'ff_pv': 1.0,
           'ff_sst': 1.0,
           'ff_vip': 0.0,
           'fb_pyr': 1.0,
           'fb_pv': 0.0,
           'fb_sst': 0.0,
           'fb_vip': 1.0},
    'L23': {'ff_pyr': 0,
            'ff_pv': 0,
            'ff_sst': 0,
            'ff_vip': 0,
            'fb_pyr': 1.0,
            'fb_pv': 0.0,
            'fb_sst': 0.0,
            'fb_vip': 1.0},
    'L5': {'ff_pyr': 0,
           'ff_pv': 0,
           'ff_sst': 0,
           'ff_vip': 0,
           'fb_pyr': 1.0,
           'fb_pv': 0.0,
           'fb_sst': 0.0,
           'fb_vip': 1.0},
    'L6': {'ff_pyr': 0.7,
           'ff_pv': 1.0,
           'ff_sst': 1.0,
           'ff_vip': 0.0,
           'fb_pyr': 1.0,
           'fb_pv': 0.0,
           'fb_sst': 0.0,
           'fb_vip': 1.0}
    }

syn_input_plast = {
    'L4': {'ff_pyr': 'reinit',
           'ff_pv': 'static',
           'ff_sst': 'static',
           'ff_vip': 'static',
           'fb_pyr': 'stdp',
           'fb_pv': 'stdp',
           'fb_sst': 'static',
           'fb_vip': 'static'},
    'L23': {'ff_pyr': 'reinit',
            'ff_pv': 'static',
            'ff_sst': 'static',
            'ff_vip': 'static',
            'fb_pyr': 'stdp',
            'fb_pv': 'stdp',
            'fb_sst': 'static',
            'fb_vip': 'static'},
    'L5': {'ff_pyr': 'reinit',
           'ff_pv': 'static',
           'ff_sst': 'static',
           'ff_vip': 'static',
           'fb_pyr': 'stdp',
           'fb_pv': 'stdp',
           'fb_sst': 'static',
           'fb_vip': 'static'},
    'L6': {'ff_pyr': 'reinit',
           'ff_pv': 'static',
           'ff_sst': 'static',
           'ff_vip': 'static',
           'fb_pyr': 'stdp',
           'fb_pv': 'stdp',
           'fb_sst': 'static',
           'fb_vip': 'static'}
    }

syn_intra_prob = {
    'L23': {
        'pyr_pyr': 0.101,
        'pyr_pv': 0.135,
        'pyr_sst': 0.135,
        'pyr_vip': 0.135,
        'pv_pyr': 1.0, #TODO
        'pv_pv': 1.0, #TODO
        'pv_sst': 0.0,
        'pv_vip': 0.0,
        'sst_pyr': 1.0, #TODO
        'sst_pv': 0.9, #TODO
        'sst_sst': 0,
        'sst_vip': 0.9, #TODO
        'vip_pyr': 0,
        'vip_pv': 0,
        'vip_sst': 0.65, #TODO
        'vip_vip': 0},
    'L4': {
        'pyr_pyr': 0.050,
        'pyr_pv': 0.079,
        'pyr_sst': 0.079,
        'pyr_vip': 0.079,
        'pv_pyr': 1.0, #TODO 0.60
        'pv_pv': 1.0, #TODO 0.50
        'pv_sst': 0.0,
        'pv_vip': 0.0,
        'sst_pyr': 1.0, #TODO 0.55
        'sst_pv': 0.9, #TODO 0.60
        'sst_sst': 0,
        'sst_vip': 0.9, #TODO 0.45
        'vip_pyr': 0,
        'vip_pv': 0,
        'vip_sst': 0.65, #TODO 0.50
        'vip_vip': 0},
    'L5': {
        'pyr_pyr': 0.083,
        'pyr_pv': 0.060,
        'pyr_sst': 0.060,
        'pyr_vip': 0.060,
        'pv_pyr': 1.0, #TODO
        'pv_pv': 1.0, #TODO
        'pv_sst': 0.0,
        'pv_vip': 0.0,
        'sst_pyr': 1.0, #TODO
        'sst_pv': 0.9, #TODO
        'sst_sst': 0,
        'sst_vip': 0.9, #TODO
        'vip_pyr': 0,
        'vip_pv': 0,
        'vip_sst': 0.65, #TODO
        'vip_vip': 0},
    'L6': {
        'pyr_pyr': 0.040,
        'pyr_pv': 0.066,
        'pyr_sst': 0.066,
        'pyr_vip': 0.066,
        'pv_pyr': 1.0, #TODO
        'pv_pv': 1.0, #TODO
        'pv_sst': 0.0,
        'pv_vip': 0.0,
        'sst_pyr': 1.0, #TODO
        'sst_pv': 0.9, #TODO
        'sst_sst': 0,
        'sst_vip': 0.9, #TODO
        'vip_pyr': 0,
        'vip_pv': 0,
        'vip_sst': 0.65, #TODO
        'vip_vip': 0},
    }

syn_intra_plast = {
    'L23': {
        'pyr_pyr': 'stdp',
        'pyr_pv': 'static',
        'pyr_sst': 'static',
        'pyr_vip': 'static',
        'pv_pyr': 'istdp',
        'pv_pv': 'static',
        'sst_pv': 'altadp',
        'sst_pyr': 'istdp',
        'sst_vip': 'static',
        'vip_sst': 'static'},
    'L4': {
        'pyr_pyr': 'stdp',
        'pyr_pv': 'static',
        'pyr_sst': 'static',
        'pyr_vip': 'static',
        'pv_pyr': 'istdp',
        'pv_pv': 'static',
        'sst_pv': 'altadp',
        'sst_pyr': 'istdp',
        'sst_vip': 'static',
        'vip_sst': 'static'},
    'L5': {
        'pyr_pyr': 'stdp',
        'pyr_pv': 'static',
        'pyr_sst': 'static',
        'pyr_vip': 'static',
        'pv_pyr': 'istdp',
        'pv_pv': 'static',
        'sst_pv': 'altadp',
        'sst_pyr': 'istdp',
        'sst_vip': 'static',
        'vip_sst': 'static'},
    'L6': {
        'pyr_pyr': 'stdp',
        'pyr_pv': 'static',
        'pyr_sst': 'static',
        'pyr_vip': 'static',
        'pv_pyr': 'istdp',
        'pv_pv': 'static',
        'sst_pv': 'altadp',
        'sst_pyr': 'istdp',
        'sst_vip': 'static',
        'vip_sst': 'static'},
    }

# The dictionaries below contain the parameters for each case, as defined above
syn_model_vals = {
    'static': {
        'ff_pyr': {
            'gain_syn': 1*mA,
            'weight': 3,
            'w_plast': 1,
            'delay': 0*ms},
        'ff_pv': {
            'gain_syn': 1*mA,
            'weight': 3,
            'w_plast': 1,
            'delay': 0*ms},
        'ff_sst': {
            'gain_syn': 1*mA,
            'weight': 3,
            'w_plast': 1,
            'delay': 0*ms},
        'ff_vip': {
            'gain_syn': 1*mA,
            'weight': 3,
            'w_plast': 1,
            'delay': 0*ms},
        'fb_pyr': {
            'gain_syn': 1*mA,
            'weight': 3,
            'w_plast': 1,
            'delay': 0*ms},
        'fb_pv': {
            'gain_syn': 1*mA,
            'weight': 3,
            'w_plast': 1,
            'delay': 0*ms},
        'fb_sst': {
            'gain_syn': 1*mA,
            'weight': 3,
            'w_plast': 1,
            'delay': 0*ms},
        'fb_vip': {
            'gain_syn': 1*mA,
            'weight': 3,
            'w_plast': 1,
            'delay': 0*ms},
        'pyr_pyr': {
            'gain_syn': 1*mA,
            'weight': 1,
            'w_plast': 1,
            'delay': 4*ms},
        'pyr_pv': {
            'gain_syn': 1*mA,
            'weight': 3,
            'w_plast': 1,
            'delay': 0*ms},
        'pyr_sst': {
            'gain_syn': 1*mA,
            'weight': 3,
            'w_plast': 1,
            'delay': 0*ms},
        'pyr_vip': {
            'gain_syn': 1*mA,
            'weight': 3,
            'w_plast': 1,
            'delay': 0*ms},
        'pv_pyr': {
            'gain_syn': 1*mA,
            'weight': -2,
            'w_plast': 1,
            'delay': 0*ms},
        'pv_pv': {
            'gain_syn': 1*mA,
            'weight': -1,
            'w_plast': 1,
            'delay': 0*ms},
        'sst_pyr': {
            'gain_syn': 1*mA,
            'weight': -2,
            'w_plast': 1,
            'delay': 0*ms},
        'sst_pv': {
            'gain_syn': 1*mA,
            'weight': -2,
            'w_plast': 1,
            'delay': 0*ms},
        'sst_vip': {
            'gain_syn': 1*mA,
            'weight': -1,
            'w_plast': 1,
            'delay': 0*ms},
        'vip_sst': {
            'gain_syn': 1*mA,
            'weight': -1,
            'w_plast': 1,
            'gain_syn': 1*mA,
            'delay': 0*ms},
        },
    'reinit': {
        'ff_pyr': {
            'gain_syn': 1*mA,
            'delay': 0*ms,
            'pre_decay_numerator': 243,
            'post_decay_numerator': 247,
            'A_max': lambda n_bits: 2**n_bits - 1,
            'rand_num_bits': lambda n_bits: n_bits,
            'rand_num_bits_syn': lambda n_bits: n_bits,
            'weight': 1,
            'w_plast': 2,
            'w_max': lambda n_bits: 2**(n_bits - 1) - 1,
            'dApre': 15,
            'stdp_thres': 1}
        },
    'istdp': {
        'pv_pyr': {
            'gain_syn': 1*mA,
            'delay': 0*ms,
            'pre_decay_numerator': 243,
            'post_decay_numerator': 247,
            'A_max': lambda n_bits: 2**n_bits - 1,
            'rand_num_bits': lambda n_bits: n_bits,
            'rand_num_bits_syn': lambda n_bits: n_bits,
            'weight': -1,
            'w_plast': 1,
            'w_max': lambda n_bits: 2**(n_bits - 1) - 1,
            'variance_th': 4,
            'dApre': 15,
            'stdp_thres': 1},
        'sst_pyr': {
            'gain_syn': 1*mA,
            'delay': 0*ms,
            'pre_decay_numerator': 243,
            'post_decay_numerator': 247,
            'A_max': lambda n_bits: 2**n_bits - 1,
            'rand_num_bits': lambda n_bits: n_bits,
            'rand_num_bits_syn': lambda n_bits: n_bits,
            'weight': -1,
            'w_plast': 1,
            'w_max': lambda n_bits: 2**(n_bits - 1) - 1,
            'variance_th': 4,
            'dApre': 15,
            'stdp_thres': 1},
        },
    'adp': {
        'pv_pyr': {
            'gain_syn': 1*mA,
            'delay': 0*ms,
            'pre_decay_numerator': 243,
            'post_decay_numerator': 247,
            'weight': -1,
            'w_plast': 1,
            'w_max': lambda n_bits: 2**(n_bits - 1) - 1,
            'variance_th': 0.50,
            'dApre': 15,
            'stdp_thres': 1},
        'sst_pyr': {
            'gain_syn': 1*mA,
            'delay': 0*ms,
            'pre_decay_numerator': 243,
            'post_decay_numerator': 247,
            'rand_num_bits': lambda n_bits: n_bits,
            'weight': -1,
            'w_plast': 1,
            'w_max': lambda n_bits: 2**(n_bits - 1) - 1,
            'variance_th': 0.50,
            'dApre': 15,
            'stdp_thres': 1},
        },
    'altadp': {
        'sst_pv': {
            'gain_syn': 1*mA,
            'delay': 0*ms,
            'pre_decay_numerator': 243,
            'post_decay_numerator': 247,
            'weight': -1,
            'w_plast': 1,
            'w_max': lambda n_bits: 2**(n_bits - 1) - 1,
            'inh_learning_rate': 0.01,
            'dApre': 15,
            'stdp_thres': 1
            }
        },
    'stdp': {
        'ff_pyr': {
            'gain_syn': 1*mA,
            'delay': 0*ms,
            'pre_decay_numerator': 243,
            'post_decay_numerator': 247,
            'A_max': lambda n_bits: 2**n_bits - 1,
            'rand_num_bits': lambda n_bits: n_bits,
            'rand_num_bits_syn': lambda n_bits: n_bits,
            'weight': 1,
            'w_plast': 2,
            'w_max': lambda n_bits: 2**(n_bits - 1) - 1,
            'dApre': 15,
            'stdp_thres': 1
            },
        'ff_pv': {
            'gain_syn': 1*mA,
            'delay': 0*ms,
            'pre_decay_numerator': 243,
            'post_decay_numerator': 247,
            'A_max': lambda n_bits: 2**n_bits - 1,
            'rand_num_bits': lambda n_bits: n_bits,
            'rand_num_bits_syn': lambda n_bits: n_bits,
            'weight': 1,
            'w_plast': 2,
            'w_max': lambda n_bits: 2**(n_bits - 1) - 1,
            'dApre': 15,
            'stdp_thres': 1
            },
        'fb_pyr': {
            'gain_syn': 1*mA,
            'delay': 0*ms,
            'pre_decay_numerator': 243,
            'post_decay_numerator': 247,
            'A_max': lambda n_bits: 2**n_bits - 1,
            'rand_num_bits': lambda n_bits: n_bits,
            'rand_num_bits_syn': lambda n_bits: n_bits,
            'weight': 1,
            'w_plast': 2,
            'w_max': lambda n_bits: 2**(n_bits - 1) - 1,
            'dApre': 15,
            'stdp_thres': 1
            },
        'fb_pv': {
            'gain_syn': 1*mA,
            'delay': 0*ms,
            'pre_decay_numerator': 243,
            'post_decay_numerator': 247,
            'A_max': lambda n_bits: 2**n_bits - 1,
            'rand_num_bits': lambda n_bits: n_bits,
            'rand_num_bits_syn': lambda n_bits: n_bits,
            'weight': 1,
            'w_plast': 2,
            'w_max': lambda n_bits: 2**(n_bits - 1) - 1,
            'dApre': 15,
            'stdp_thres': 1
            },
        'pyr_pyr': {
            'gain_syn': 1*mA,
            'delay': 4*ms,
            'pre_decay_numerator': 243,
            'post_decay_numerator': 247,
            'A_max': lambda n_bits: 2**n_bits - 1,
            'rand_num_bits': lambda n_bits: n_bits,
            'rand_num_bits_syn': lambda n_bits: n_bits,
            'weight': 1,
            'w_plast': 4,
            'w_max': lambda n_bits: 2**(n_bits - 1) - 1,
            'dApre': 15,
            'stdp_thres': 1
            },
        },
    'redsymstdp': {
        'ff_pyr': {
            'gain_syn': 1*mA,
            'delay': 0*ms,
            'pre_decay_numerator': 243,
            'post_decay_numerator': 247,
            'A_max': lambda n_bits: 2**n_bits - 1,
            'rand_num_bits': lambda n_bits: n_bits,
            'rand_num_bits_syn': lambda n_bits: n_bits,
            'weight': 1,
            'w_plast': 2,
            'w_max': lambda n_bits: 2**(n_bits - 1) - 1,
            'dApre': 15,
            'stdp_thres': 1
            },
        'ff_pv': {
            'gain_syn': 1*mA,
            'delay': 0*ms,
            'pre_decay_numerator': 243,
            'post_decay_numerator': 247,
            'A_max': lambda n_bits: 2**n_bits - 1,
            'rand_num_bits': lambda n_bits: n_bits,
            'rand_num_bits_syn': lambda n_bits: n_bits,
            'weight': 1,
            'w_plast': 2,
            'w_max': lambda n_bits: 2**(n_bits - 1) - 1,
            'dApre': 15,
            'stdp_thres': 1
            },
        'fb_pyr': {
            'gain_syn': 1*mA,
            'delay': 0*ms,
            'pre_decay_numerator': 243,
            'post_decay_numerator': 247,
            'A_max': lambda n_bits: 2**n_bits - 1,
            'rand_num_bits': lambda n_bits: n_bits,
            'rand_num_bits_syn': lambda n_bits: n_bits,
            'weight': 1,
            'w_plast': 2,
            'w_max': lambda n_bits: 2**(n_bits - 1) - 1,
            'dApre': 15,
            'stdp_thres': 1
            },
        'fb_pv': {
            'gain_syn': 1*mA,
            'delay': 0*ms,
            'pre_decay_numerator': 243,
            'post_decay_numerator': 247,
            'A_max': lambda n_bits: 2**n_bits - 1,
            'rand_num_bits': lambda n_bits: n_bits,
            'rand_num_bits_syn': lambda n_bits: n_bits,
            'weight': 1,
            'w_plast': 2,
            'w_max': lambda n_bits: 2**(n_bits - 1) - 1,
            'dApre': 15,
            'stdp_thres': 1
            },
        'pyr_pyr': {
            'gain_syn': 1*mA,
            'delay': 4*ms,
            'pre_decay_numerator': 243,
            'post_decay_numerator': 247,
            'A_max': lambda n_bits: 2**n_bits - 1,
            'rand_num_bits': lambda n_bits: n_bits,
            'rand_num_bits_syn': lambda n_bits: n_bits,
            'weight': 1,
            'w_plast': 4,
            'w_max': lambda n_bits: 2**(n_bits - 1) - 1,
            'dApre': 15,
            'stdp_thres': 1
            },
        },
    'hredsymstdp': {
        'ff_pyr': {
            'gain_syn': 1*mA,
            'delay': 0*ms,
            'pre_decay_numerator': 243,
            'post_decay_numerator': 247,
            'A_max': lambda n_bits: 2**n_bits - 1,
            'rand_num_bits': lambda n_bits: n_bits,
            'rand_num_bits_syn': lambda n_bits: n_bits,
            'weight': 1,
            'w_plast': 2,
            'w_max': lambda n_bits: 2**(n_bits - 1) - 1,
            'max_summed_w': 15,
            'dApre': 15,
            'stdp_thres': 1
            },
        'pyr_pyr': {
            'gain_syn': 1*mA,
            'delay': 4*ms,
            'pre_decay_numerator': 243,
            'post_decay_numerator': 247,
            'A_max': lambda n_bits: 2**n_bits - 1,
            'rand_num_bits': lambda n_bits: n_bits,
            'rand_num_bits_syn': lambda n_bits: n_bits,
            'weight': 1,
            'w_plast': 4,
            'w_max': lambda n_bits: 2**(n_bits - 1) - 1,
            'max_summed_w': 15,
            'dApre': 15,
            'stdp_thres': 1
            },
        }
    }

# Values of type string must correspond to a key in model_vals (except variable)
syn_sample_vars = {
    'static': {
        'ff_pyr': {
            'weight': {
                'unit': 1, 'dist_type': 'normal',
                'sign': lambda weight: np.sign(weight), 'min': 1,
                'max': lambda n_bits: 2**(n_bits - 1) - 1},
            },
        'ff_pv': {
            'weight': {
                'unit': 1, 'dist_type': 'normal',
                'sign': lambda weight: np.sign(weight), 'min': 1,
                'max': lambda n_bits: 2**(n_bits - 1) - 1},
            },
        'ff_sst': {
            'weight': {
                'unit': 1, 'dist_type': 'normal',
                'sign': lambda weight: np.sign(weight), 'min': 1,
                'max': lambda n_bits: 2**(n_bits - 1) - 1},
            },
        'ff_vip': {
            'weight': {
                'unit': 1, 'dist_type': 'normal',
                'sign': lambda weight: np.sign(weight), 'min': 1,
                'max': lambda n_bits: 2**(n_bits - 1) - 1},
            },
        'fb_pyr': {
            'weight': {
                'unit': 1, 'dist_type': 'normal',
                'sign': lambda weight: np.sign(weight), 'min': 1,
                'max': lambda n_bits: 2**(n_bits - 1) - 1},
            },
        'fb_pv': {
            'weight': {
                'unit': 1, 'dist_type': 'normal',
                'sign': lambda weight: np.sign(weight), 'min': 1,
                'max': lambda n_bits: 2**(n_bits - 1) - 1},
            },
        'fb_sst': {
            'weight': {
                'unit': 1, 'dist_type': 'normal',
                'sign': lambda weight: np.sign(weight), 'min': 1,
                'max': lambda n_bits: 2**(n_bits - 1) - 1},
            },
        'fb_vip': {
            'weight': {
                'unit': 1, 'dist_type': 'normal',
                'sign': lambda weight: np.sign(weight), 'min': 1,
                'max': lambda n_bits: 2**(n_bits - 1) - 1},
            },
        'pyr_pyr': {
            'weight': {
                'unit': 1, 'dist_type': 'normal',
                'sign': lambda weight: np.sign(weight), 'min': 1,
                'max': lambda n_bits: 2**(n_bits - 1) - 1},
            'delay': {
                'unit': 1*ms, 'dist_type': 'uniform', 'sign': 1, 'min': 0, 'max': 20},
            },
        'pyr_pv': {
                'weight': {
                'unit': 1, 'dist_type': 'normal',
                'sign': lambda weight: np.sign(weight), 'min': 1,
                'max': lambda n_bits: 2**(n_bits - 1) - 1},
            },
        'pyr_sst': {
                'weight': {
                'unit': 1, 'dist_type': 'normal',
                'sign': lambda weight: np.sign(weight), 'min': 1,
                'max': lambda n_bits: 2**(n_bits - 1) - 1},
            },
        'pyr_vip': {
                'weight': {
                'unit': 1, 'dist_type': 'normal',
                'sign': lambda weight: np.sign(weight), 'min': 1,
                'max': lambda n_bits: 2**(n_bits - 1) - 1},
            },
        'pv_pyr': {
                'weight': {
                'unit': 1, 'dist_type': 'normal',
                'sign': lambda weight: np.sign(weight), 'min': 1,
                'max': lambda n_bits: 2**(n_bits - 1) - 1},
            },
        'pv_pv': {
                'weight': {
                'unit': 1, 'dist_type': 'normal',
                'sign': lambda weight: np.sign(weight), 'min': 1,
                'max': lambda n_bits: 2**(n_bits - 1) - 1},
            },
        'sst_pv': {
                'weight': {
                'unit': 1, 'dist_type': 'normal',
                'sign': lambda weight: np.sign(weight), 'min': 1,
                'max': lambda n_bits: 2**(n_bits - 1) - 1},
            },
        'sst_pyr': {
                'weight': {
                'unit': 1, 'dist_type': 'normal',
                'sign': lambda weight: np.sign(weight), 'min': 1,
                'max': lambda n_bits: 2**(n_bits - 1) - 1},
            },
        'sst_vip': {
                'weight': {
                'unit': 1, 'dist_type': 'normal',
                'sign': lambda weight: np.sign(weight), 'min': 1,
                'max': lambda n_bits: 2**(n_bits - 1) - 1},
            },
        'vip_sst': {
                'weight': {
                'unit': 1, 'dist_type': 'normal',
                'sign': lambda weight: np.sign(weight), 'min': 1,
                'max': lambda n_bits: 2**(n_bits - 1) - 1},
            },
        },
    'reinit': {
        'ff_pyr': {
            'w_plast': {
                'unit': 1, 'dist_type': 'normal', 'sign': 1, 'min': 1,
                'max': lambda w_max: w_max},
            'pre_decay_numerator': {
                'unit': 1, 'dist_type': 'normal', 'sign': 1, 'min': 0,
                'max': lambda max_numerator: max_numerator},
            'post_decay_numerator': {
                'unit': 1, 'dist_type': 'normal', 'sign': 1, 'min': 0,
                'max': lambda max_numerator: max_numerator},
            }
        },
    'istdp': {
        'pv_pyr': {
            'w_plast': {
                'unit': 1, 'dist_type': 'normal', 'sign': 1, 'min': 1,
                'max': lambda w_max: w_max},
            'pre_decay_numerator': {
                'unit': 1, 'dist_type': 'normal', 'sign': 1, 'min': 0,
                'max': lambda max_numerator: max_numerator},
            'post_decay_numerator': {
                'unit': 1, 'dist_type': 'normal', 'sign': 1, 'min': 0,
                'max': lambda max_numerator: max_numerator},
            },
        'sst_pyr': {
            'w_plast': {
                'unit': 1, 'dist_type': 'normal', 'sign': 1, 'min': 1,
                'max': lambda w_max: w_max},
            'pre_decay_numerator': {
                'unit': 1, 'dist_type': 'normal', 'sign': 1, 'min': 0,
                'max': lambda max_numerator: max_numerator},
            'post_decay_numerator': {
                'unit': 1, 'dist_type': 'normal', 'sign': 1, 'min': 0,
                'max': lambda max_numerator: max_numerator},
            }
        },
    'adp': {
        'sst_pv': {
            'w_plast': {
                'unit': 1, 'dist_type': 'normal', 'sign': 1, 'min': 1,
                'max': lambda w_max: w_max},
            }
        },
    'altadp': {
        'sst_pv': {
            'w_plast': {
                'unit': 1, 'dist_type': 'normal', 'sign': 1, 'min': 1,
                'max': lambda w_max: w_max},
            }
        },
    'stdp': {
        'ff_pyr': {
            'w_plast': {
                'unit': 1, 'dist_type': 'normal', 'sign': 1, 'min': 1,
                'max': lambda w_max: w_max},
            'pre_decay_numerator': {
                'unit': 1, 'dist_type': 'normal', 'sign': 1, 'min': 0,
                'max': lambda max_numerator: max_numerator},
            'post_decay_numerator': {
                'unit': 1, 'dist_type': 'normal', 'sign': 1, 'min': 0,
                'max': lambda max_numerator: max_numerator},
            },
        'ff_pv': {
            'w_plast': {
                'unit': 1, 'dist_type': 'normal', 'sign': 1, 'min': 1,
                'max': lambda w_max: w_max},
            'pre_decay_numerator': {
                'unit': 1, 'dist_type': 'normal', 'sign': 1, 'min': 0,
                'max': lambda max_numerator: max_numerator},
            'post_decay_numerator': {
                'unit': 1, 'dist_type': 'normal', 'sign': 1, 'min': 0,
                'max': lambda max_numerator: max_numerator},
            },
        'fb_pyr': {
            'w_plast': {
                'unit': 1, 'dist_type': 'normal', 'sign': 1, 'min': 1,
                'max': lambda w_max: w_max},
            'pre_decay_numerator': {
                'unit': 1, 'dist_type': 'normal', 'sign': 1, 'min': 0,
                'max': lambda max_numerator: max_numerator},
            'post_decay_numerator': {
                'unit': 1, 'dist_type': 'normal', 'sign': 1, 'min': 0,
                'max': lambda max_numerator: max_numerator},
            },
        'fb_pv': {
            'w_plast': {
                'unit': 1, 'dist_type': 'normal', 'sign': 1, 'min': 1,
                'max': lambda w_max: w_max},
            'pre_decay_numerator': {
                'unit': 1, 'dist_type': 'normal', 'sign': 1, 'min': 0,
                'max': lambda max_numerator: max_numerator},
            'post_decay_numerator': {
                'unit': 1, 'dist_type': 'normal', 'sign': 1, 'min': 0,
                'max': lambda max_numerator: max_numerator},
            },
        'pyr_pyr': {
            'w_plast': {
                'unit': 1, 'dist_type': 'normal', 'sign': 1, 'min': 1,
                'max': lambda w_max: w_max},
            'delay': {
                'unit': 1*ms, 'dist_type': 'uniform', 'sign': 1, 'min': 0, 'max': 20},
            'pre_decay_numerator': {
                'unit': 1, 'dist_type': 'normal', 'sign': 1, 'min': 0,
                'max': lambda max_numerator: max_numerator},
            'post_decay_numerator': {
                'unit': 1, 'dist_type': 'normal', 'sign': 1, 'min': 0,
                'max': lambda max_numerator: max_numerator},
            }
        },
    'redsymstdp': {
        'ff_pyr': {
            'w_plast': {
                'unit': 1, 'dist_type': 'normal', 'sign': 1, 'min': 1,
                'max': lambda w_max: w_max},
            'pre_decay_numerator': {
                'unit': 1, 'dist_type': 'normal', 'sign': 1, 'min': 0,
                'max': lambda max_numerator: max_numerator},
            'post_decay_numerator': {
                'unit': 1, 'dist_type': 'normal', 'sign': 1, 'min': 0,
                'max': lambda max_numerator: max_numerator},
            },
        'ff_pv': {
            'w_plast': {
                'unit': 1, 'dist_type': 'normal', 'sign': 1, 'min': 1,
                'max': lambda w_max: w_max},
            'pre_decay_numerator': {
                'unit': 1, 'dist_type': 'normal', 'sign': 1, 'min': 0,
                'max': lambda max_numerator: max_numerator},
            'post_decay_numerator': {
                'unit': 1, 'dist_type': 'normal', 'sign': 1, 'min': 0,
                'max': lambda max_numerator: max_numerator},
            },
        'fb_pyr': {
            'w_plast': {
                'unit': 1, 'dist_type': 'normal', 'sign': 1, 'min': 1,
                'max': lambda w_max: w_max},
            'pre_decay_numerator': {
                'unit': 1, 'dist_type': 'normal', 'sign': 1, 'min': 0,
                'max': lambda max_numerator: max_numerator},
            'post_decay_numerator': {
                'unit': 1, 'dist_type': 'normal', 'sign': 1, 'min': 0,
                'max': lambda max_numerator: max_numerator},
            },
        'fb_pv': {
            'w_plast': {
                'unit': 1, 'dist_type': 'normal', 'sign': 1, 'min': 1,
                'max': lambda w_max: w_max},
            'pre_decay_numerator': {
                'unit': 1, 'dist_type': 'normal', 'sign': 1, 'min': 0,
                'max': lambda max_numerator: max_numerator},
            'post_decay_numerator': {
                'unit': 1, 'dist_type': 'normal', 'sign': 1, 'min': 0,
                'max': lambda max_numerator: max_numerator},
            },
        'pyr_pyr': {
            'w_plast': {
                'unit': 1, 'dist_type': 'normal', 'sign': 1, 'min': 1,
                'max': lambda w_max: w_max},
            'delay': {
                'unit': 1*ms, 'dist_type': 'uniform', 'sign': 1, 'min': 0, 'max': 20},
            'pre_decay_numerator': {
                'unit': 1, 'dist_type': 'normal', 'sign': 1, 'min': 0,
                'max': lambda max_numerator: max_numerator},
            'post_decay_numerator': {
                'unit': 1, 'dist_type': 'normal', 'sign': 1, 'min': 0,
                'max': lambda max_numerator: max_numerator},
            }
        },
    'hredsymstdp': {
        'ff_pyr': {
            'w_plast': {
                'unit': 1, 'dist_type': 'normal', 'sign': 1, 'min': 1,
                'max': lambda w_max: w_max},
            'pre_decay_numerator': {
                'unit': 1, 'dist_type': 'normal', 'sign': 1, 'min': 0,
                'max': lambda max_numerator: max_numerator},
            'post_decay_numerator': {
                'unit': 1, 'dist_type': 'normal', 'sign': 1, 'min': 0,
                'max': lambda max_numerator: max_numerator},
            },
        'pyr_pyr': {
            'w_plast': {
                'unit': 1, 'dist_type': 'normal', 'sign': 1, 'min': 1,
                'max': lambda w_max: w_max},
            'delay': {
                'unit': 1*ms, 'dist_type': 'uniform', 'sign': 1, 'min': 0, 'max': 20},
            'pre_decay_numerator': {
                'unit': 1, 'dist_type': 'normal', 'sign': 1, 'min': 0,
                'max': lambda max_numerator: max_numerator},
            'post_decay_numerator': {
                'unit': 1, 'dist_type': 'normal', 'sign': 1, 'min': 0,
                'max': lambda max_numerator: max_numerator},
            }
        }
    }

reinit_vars = {
    'ff_pyr': {
        're_init_dt': 60000*ms
        }
    }

# Dictionaries of neuronal populations
neu_pop = {
    'L23': {
        'n_exc': 20683,
        'ei_ratio': 3.545,
        'inh_ratio': {
            'pv_cells': .37,
            'sst_cells': .20,
            'vip_cells': .43
            },
        'num_inputs': {
            'pyr_cells': 4,
            'pv_cells': 4,
            'sst_cells': 3,
            'vip_cells': 2
            },
        },
    'L4': {
        'n_exc': 21915,
        'ei_ratio': 3.999, # TODO 1.6 in case I need more inhibition?
        # TODO use commented values if I want only PVs
        'inh_ratio': {
            'pv_cells': .68,#1
            'sst_cells': .20,#.02,
            'vip_cells': .12,#.02
            },
        'num_inputs': {
            'pyr_cells': 4,
            'pv_cells': 4,
            'sst_cells': 3,
            'vip_cells': 2
            },
        },
    'L5': {
        'n_exc': 4850,
        'ei_ratio': 4.55,
        'inh_ratio': {
            'pv_cells': .52,
            'sst_cells': .37,
            'vip_cells': .11
            },
        'num_inputs': {
            'pyr_cells': 4,
            'pv_cells': 4,
            'sst_cells': 3,
            'vip_cells': 2
            },
        },
    'L6': {
        'n_exc': 14395,
        'ei_ratio': 4.882,
        'inh_ratio': {
            'pv_cells': .49,
            'sst_cells': .38,
            'vip_cells': .13
            },
        'num_inputs': {
            'pyr_cells': 4,
            'pv_cells': 4,
            'sst_cells': 3,
            'vip_cells': 2
            },
        },
    }

neu_pop_plast = {
    'pyr_cells': 'adapt',
    'pv_cells': 'adapt',
    'sst_cells': 'static',
    'vip_cells': 'static'
    }

neu_model_vals = {
    'static': {
        'pyr_cells': {
            'tau': 20*ms,
            'decay_numerator': 244,
            'refrac_tau': 2*ms,
            'refrac_decay_numerator': 154,
            'tausyn': 5*ms,
            'syn_decay_numerator': 213,
            'rand_num_bits': lambda n_bits: n_bits,
            'Vm': lambda n_bits: (2**n_bits - 1)/5*mV,
            'Vrest': lambda n_bits: (2**n_bits - 1)/5*mV,
            'Vm_max': lambda n_bits: (2**n_bits - 1),
            'Vthr': lambda n_bits: (2**n_bits - 1)*mV,
            'I_min': lambda n_bits: -2**(n_bits-1)*mA,
            'I_max': lambda n_bits: (2**(n_bits-1) - 1)*mA,
            'Vm_noise': 0*mV
            },
        'pv_cells': {
            'tau': 20*ms,
            'decay_numerator': 244,
            'refrac_tau': 2*ms,
            'refrac_decay_numerator': 154,
            'tausyn': 5*ms,
            'syn_decay_numerator': 213,
            'rand_num_bits': lambda n_bits: n_bits,
            'Vm': lambda n_bits: (2**n_bits - 1)/5*mV,
            'Vrest': lambda n_bits: (2**n_bits - 1)/5*mV,
            'Vm_max': lambda n_bits: (2**n_bits - 1),
            'Vthr': lambda n_bits: (2**n_bits - 1)*mV,
            'I_min': lambda n_bits: -2**(n_bits-1)*mA,
            'I_max': lambda n_bits: (2**(n_bits-1) - 1)*mA,
            'Vm_noise': 0*mV},
        'sst_cells': {
            'tau': 20*ms,
            'decay_numerator': 244,
            'refrac_tau': 2*ms,
            'refrac_decay_numerator': 154,
            'tausyn': 5*ms,
            'syn_decay_numerator': 213,
            'rand_num_bits': lambda n_bits: n_bits,
            'Vm': lambda n_bits: (2**n_bits - 1)/5*mV,
            'Vrest': lambda n_bits: (2**n_bits - 1)/5*mV,
            'Vm_max': lambda n_bits: (2**n_bits - 1),
            'Vthr': lambda n_bits: (2**n_bits - 1)*mV,
            'I_min': lambda n_bits: -2**(n_bits-1)*mA,
            'I_max': lambda n_bits: (2**(n_bits-1) - 1)*mA,
            'Vm_noise': 0*mV},
        'vip_cells': {
            'tau': 20*ms,
            'decay_numerator': 244,
            'refrac_tau': 2*ms,
            'refrac_decay_numerator': 154,
            'tausyn': 5*ms,
            'syn_decay_numerator': 213,
            'rand_num_bits': lambda n_bits: n_bits,
            'Vm': lambda n_bits: (2**n_bits - 1)/5*mV,
            'Vrest': lambda n_bits: (2**n_bits - 1)/5*mV,
            'Vm_max': lambda n_bits: (2**n_bits - 1),
            'Vthr': lambda n_bits: (2**n_bits - 1)*mV,
            'I_min': lambda n_bits: -2**(n_bits-1)*mA,
            'I_max': lambda n_bits: (2**(n_bits-1) - 1)*mA,
            'Vm_noise': 0*mV},
        },
    'adapt': {
        'pyr_cells': {
            'tau': 20*ms,
            'decay_numerator': 244,
            'refrac_tau': 2*ms,
            'refrac_decay_numerator': 154,
            'tausyn': 5*ms,
            'syn_decay_numerator': 213,
            'rand_num_bits': lambda n_bits: n_bits,
            'Vm': lambda n_bits: (2**n_bits - 1)/5*mV,
            'Vrest': lambda n_bits: (2**n_bits - 1)/5*mV,
            'Vm_max': lambda n_bits: (2**n_bits - 1),
            'Vthr': lambda n_bits: (2**n_bits - 1)*mV,
            'I_min': lambda n_bits: -2**(n_bits-1)*mA,
            'I_max': lambda n_bits: (2**(n_bits-1) - 1)*mA,
            'thr_min': lambda n_bits: np.ceil(2**n_bits/5)*mV,
            'thr_max': lambda n_bits: (2**n_bits - 1)*mV,
            'Vm_noise': 0*mV
            },
        'pv_cells': {
            'tau': 20*ms,
            'decay_numerator': 244,
            'refrac_tau': 2*ms,
            'refrac_decay_numerator': 154,
            'tausyn': 5*ms,
            'syn_decay_numerator': 213,
            'rand_num_bits': lambda n_bits: n_bits,
            'Vm': lambda n_bits: (2**n_bits - 1)/5*mV,
            'Vrest': lambda n_bits: (2**n_bits - 1)/5*mV,
            'Vm_max': lambda n_bits: (2**n_bits - 1),
            'Vthr': lambda n_bits: (2**n_bits - 1)*mV,
            'I_min': lambda n_bits: -2**(n_bits-1)*mA,
            'I_max': lambda n_bits: (2**(n_bits-1) - 1)*mA,
            'thr_min': lambda n_bits: np.ceil(2**n_bits/5)*mV,
            'thr_max': lambda n_bits: (2**n_bits - 1)*mV,
            'Vm_noise': 0*mV},
        'sst_cells': {
            'tau': 20*ms,
            'decay_numerator': 244,
            'refrac_tau': 2*ms,
            'refrac_decay_numerator': 154,
            'tausyn': 5*ms,
            'syn_decay_numerator': 213,
            'rand_num_bits': lambda n_bits: n_bits,
            'Vm': lambda n_bits: (2**n_bits - 1)/5*mV,
            'Vrest': lambda n_bits: (2**n_bits - 1)/5*mV,
            'Vm_max': lambda n_bits: (2**n_bits - 1),
            'Vthr': lambda n_bits: (2**n_bits - 1)*mV,
            'I_min': lambda n_bits: -2**(n_bits-1)*mA,
            'I_max': lambda n_bits: (2**(n_bits-1) - 1)*mA,
            'thr_min': lambda n_bits: np.ceil(2**n_bits/5)*mV,
            'thr_max': lambda n_bits: (2**n_bits - 1)*mV,
            'Vm_noise': 0*mV},
        'vip_cells': {
            'tau': 20*ms,
            'decay_numerator': 244,
            'refrac_tau': 2*ms,
            'refrac_decay_numerator': 154,
            'tausyn': 5*ms,
            'syn_decay_numerator': 213,
            'rand_num_bits': lambda n_bits: n_bits,
            'Vm': lambda n_bits: (2**n_bits - 1)/5*mV,
            'Vrest': lambda n_bits: (2**n_bits - 1)/5*mV,
            'Vm_max': lambda n_bits: (2**n_bits - 1),
            'Vthr': lambda n_bits: (2**n_bits - 1)*mV,
            'I_min': lambda n_bits: -2**(n_bits-1)*mA,
            'I_max': lambda n_bits: (2**(n_bits-1) - 1)*mA,
            'thr_min': lambda n_bits: np.ceil(2**n_bits/5)*mV,
            'thr_max': lambda n_bits: (2**n_bits - 1)*mV,
            'Vm_noise': 0*mV}
        },
    }

neu_sample_vars = {
    'pyr_cells': {
        'decay_numerator': {
            'unit': 1, 'dist_type': 'normal', 'sign': 1, 'min': 1, 'max': 255},
        'syn_decay_numerator': {
            'unit': 1, 'dist_type': 'normal', 'sign': 1, 'min': 1, 'max': 255},
        },
    'pv_cells': {
        'decay_numerator': {
            'unit': 1, 'dist_type': 'normal', 'sign': 1, 'min': 1, 'max': 255},
        'syn_decay_numerator': {
            'unit': 1, 'dist_type': 'normal', 'sign': 1, 'min': 1, 'max': 255},
        },
    'sst_cells': {
        'decay_numerator': {
            'unit': 1, 'dist_type': 'normal', 'sign': 1, 'min': 1, 'max': 255},
        'syn_decay_numerator': {
            'unit': 1, 'dist_type': 'normal', 'sign': 1, 'min': 1, 'max': 255},
        },
    'vip_cells': {
        'decay_numerator': {
            'unit': 1, 'dist_type': 'normal', 'sign': 1, 'min': 1, 'max': 255},
        'syn_decay_numerator': {
            'unit': 1, 'dist_type': 'normal', 'sign': 1, 'min': 1, 'max': 255},
        },
    }

# Dictionaries for interlaminar connections
syn_inter_plast = {
    'L23_L4': {
        'pyr_pyr': 'stdp',
        'pyr_pv': 'static',
        'pyr_sst': 'static',
        'pyr_vip': 'static',
        'pv_pyr': 'static',
        'pv_pv': 'static',
        'pv_sst': 'static',
        'pv_vip': 'static',
        'sst_pyr': 'static',
        'sst_pv': 'static',
        'sst_sst': 'static',
        'sst_vip': 'static',
        'vip_pyr': 'static',
        'vip_pv': 'static',
        'vip_sst': 'static',
        'vip_vip': 'static'},
    'L23_L5': {
        'pyr_pyr': 'stdp',
        'pyr_pv': 'static',
        'pyr_sst': 'static',
        'pyr_vip': 'static',
        'pv_pyr': 'static',
        'pv_pv': 'static',
        'pv_sst': 'static',
        'pv_vip': 'static',
        'sst_pyr': 'static',
        'sst_pv': 'static',
        'sst_sst': 'static',
        'sst_vip': 'static',
        'vip_pyr': 'static',
        'vip_pv': 'static',
        'vip_sst': 'static',
        'vip_vip': 'static'},
    'L23_L6': {
        'pyr_pyr': 'stdp',
        'pyr_pv': 'static',
        'pyr_sst': 'static',
        'pyr_vip': 'static',
        'pv_pyr': 'static',
        'pv_pv': 'static',
        'pv_sst': 'static',
        'pv_vip': 'static',
        'sst_pyr': 'static',
        'sst_pv': 'static',
        'sst_sst': 'static',
        'sst_vip': 'static',
        'vip_pyr': 'static',
        'vip_pv': 'static',
        'vip_sst': 'static',
        'vip_vip': 'static'},
    'L4_L23': {
        'pyr_pyr': 'stdp',
        'pyr_pv': 'static',
        'pyr_sst': 'static',
        'pyr_vip': 'static',
        'pv_pyr': 'static',
        'pv_pv': 'static',
        'pv_sst': 'static',
        'pv_vip': 'static',
        'sst_pyr': 'static',
        'sst_pv': 'static',
        'sst_sst': 'static',
        'sst_vip': 'static',
        'vip_pyr': 'static',
        'vip_pv': 'static',
        'vip_sst': 'static',
        'vip_vip': 'static'},
    'L4_L5': {
        'pyr_pyr': 'stdp',
        'pyr_pv': 'static',
        'pyr_sst': 'static',
        'pyr_vip': 'static',
        'pv_pyr': 'static',
        'pv_pv': 'static',
        'pv_sst': 'static',
        'pv_vip': 'static',
        'sst_pyr': 'static',
        'sst_pv': 'static',
        'sst_sst': 'static',
        'sst_vip': 'static',
        'vip_pyr': 'static',
        'vip_pv': 'static',
        'vip_sst': 'static',
        'vip_vip': 'static'},
    'L4_L6': {
        'pyr_pyr': 'stdp',
        'pyr_pv': 'static',
        'pyr_sst': 'static',
        'pyr_vip': 'static',
        'pv_pyr': 'static',
        'pv_pv': 'static',
        'pv_sst': 'static',
        'pv_vip': 'static',
        'sst_pyr': 'static',
        'sst_pv': 'static',
        'sst_sst': 'static',
        'sst_vip': 'static',
        'vip_pyr': 'static',
        'vip_pv': 'static',
        'vip_sst': 'static',
        'vip_vip': 'static'},
    'L5_L23': {
        'pyr_pyr': 'stdp',
        'pyr_pv': 'static',
        'pyr_sst': 'static',
        'pyr_vip': 'static',
        'pv_pyr': 'static',
        'pv_pv': 'static',
        'pv_sst': 'static',
        'pv_vip': 'static',
        'sst_pyr': 'static',
        'sst_pv': 'static',
        'sst_sst': 'static',
        'sst_vip': 'static',
        'vip_pyr': 'static',
        'vip_pv': 'static',
        'vip_sst': 'static',
        'vip_vip': 'static'},
    'L5_L4': {
        'pyr_pyr': 'stdp',
        'pyr_pv': 'static',
        'pyr_sst': 'static',
        'pyr_vip': 'static',
        'pv_pyr': 'static',
        'pv_pv': 'static',
        'pv_sst': 'static',
        'pv_vip': 'static',
        'sst_pyr': 'static',
        'sst_pv': 'static',
        'sst_sst': 'static',
        'sst_vip': 'static',
        'vip_pyr': 'static',
        'vip_pv': 'static',
        'vip_sst': 'static',
        'vip_vip': 'static'},
    'L5_L6': {
        'pyr_pyr': 'stdp',
        'pyr_pv': 'static',
        'pyr_sst': 'static',
        'pyr_vip': 'static',
        'pv_pyr': 'static',
        'pv_pv': 'static',
        'pv_sst': 'static',
        'pv_vip': 'static',
        'sst_pyr': 'static',
        'sst_pv': 'static',
        'sst_sst': 'static',
        'sst_vip': 'static',
        'vip_pyr': 'static',
        'vip_pv': 'static',
        'vip_sst': 'static',
        'vip_vip': 'static'},
    'L6_L23': {
        'pyr_pyr': 'stdp',
        'pyr_pv': 'static',
        'pyr_sst': 'static',
        'pyr_vip': 'static',
        'pv_pyr': 'static',
        'pv_pv': 'static',
        'pv_sst': 'static',
        'pv_vip': 'static',
        'sst_pyr': 'static',
        'sst_pv': 'static',
        'sst_sst': 'static',
        'sst_vip': 'static',
        'vip_pyr': 'static',
        'vip_pv': 'static',
        'vip_sst': 'static',
        'vip_vip': 'static'},
    'L6_L4': {
        'pyr_pyr': 'stdp',
        'pyr_pv': 'static',
        'pyr_sst': 'static',
        'pyr_vip': 'static',
        'pv_pyr': 'static',
        'pv_pv': 'static',
        'pv_sst': 'static',
        'pv_vip': 'static',
        'sst_pyr': 'static',
        'sst_pv': 'static',
        'sst_sst': 'static',
        'sst_vip': 'static',
        'vip_pyr': 'static',
        'vip_pv': 'static',
        'vip_sst': 'static',
        'vip_vip': 'static'},
    'L6_L5': {
        'pyr_pyr': 'stdp',
        'pyr_pv': 'static',
        'pyr_sst': 'static',
        'pyr_vip': 'static',
        'pv_pyr': 'static',
        'pv_pv': 'static',
        'pv_sst': 'static',
        'pv_vip': 'static',
        'sst_pyr': 'static',
        'sst_pv': 'static',
        'sst_sst': 'static',
        'sst_vip': 'static',
        'vip_pyr': 'static',
        'vip_pv': 'static',
        'vip_sst': 'static',
        'vip_vip': 'static'},
    }

syn_inter_prob = {
    'L23_L4': {
        'pyr_pyr': 0.008,
        'pyr_pv': 0.069,
        'pyr_sst': 0.069,
        'pyr_vip': 0.069,
        'pv_pyr': 0.006,
        'pv_pv': 0.003,
        'pv_sst': 0,
        'pv_vip': 0,
        'sst_pyr': 0.006,
        'sst_pv': 0.003,
        'sst_sst': 0,
        'sst_vip': 0.003,
        'vip_pyr': 0,
        'vip_pv': 0,
        'vip_sst': 0.003,
        'vip_vip': 0},
    'L23_L5': {
        'pyr_pyr': 0.100,
        'pyr_pv': 0.055,
        'pyr_sst': 0.055,
        'pyr_vip': 0.055,
        'pv_pyr': 0.062,
        'pv_pv': 0.027,
        'pv_sst': 0,
        'pv_vip': 0,
        'sst_pyr': 0.062,
        'sst_pv': 0.027,
        'sst_sst': 0,
        'sst_vip': 0.027,
        'vip_pyr': 0,
        'vip_pv': 0,
        'vip_sst': 0.027,
        'vip_vip': 0},
    'L23_L6': {
        'pyr_pyr': 0.016,
        'pyr_pv': 0.036,
        'pyr_sst': 0.036,
        'pyr_vip': 0.036,
        'pv_pyr': 0.007,
        'pv_pv': 0.001,
        'pv_sst': 0,
        'pv_vip': 0,
        'sst_pyr': 0.007,
        'sst_pv': 0.001,
        'sst_sst': 0,
        'sst_vip': 0.001,
        'vip_pyr': 0,
        'vip_pv': 0,
        'vip_sst': 0.001,
        'vip_vip': 0},
    'L4_L23': {
        'pyr_pyr': 0.044,
        'pyr_pv': 0.032,
        'pyr_sst': 0.032,
        'pyr_vip': 0.032,
        'pv_pyr': 0.082,
        'pv_pv': 0.052,
        'pv_sst': 0,
        'pv_vip': 0,
        'sst_pyr': 0.082,
        'sst_pv': 0.052,
        'sst_sst': 0,
        'sst_vip': 0.052,
        'vip_pyr': 0,
        'vip_pv': 0,
        'vip_sst': 0.052,
        'vip_vip': 0},
    'L4_L5': {
        'pyr_pyr': 0.051,
        'pyr_pv': 0.026,
        'pyr_sst': 0.026,
        'pyr_vip': 0.026,
        'pv_pyr': 0.006,
        'pv_pv': 0.002,
        'pv_sst': 0,
        'pv_vip': 0,
        'sst_pyr': 0.006,
        'sst_pv': 0.002,
        'sst_sst': 0,
        'sst_vip': 0.002,
        'vip_pyr': 0,
        'vip_pv': 0,
        'vip_sst': 0.002,
        'vip_vip': 0},
    'L4_L6': {
        'pyr_pyr': 0.021,
        'pyr_pv': 0.003,
        'pyr_sst': 0.003,
        'pyr_vip': 0.003,
        'pv_pyr': 0.017,
        'pv_pv': 0.001,
        'pv_sst': 0,
        'pv_vip': 0,
        'sst_pyr': 0.017,
        'sst_pv': 0.001,
        'sst_sst': 0,
        'sst_vip': 0.001,
        'vip_pyr': 0,
        'vip_pv': 0,
        'vip_sst': 0.001,
        'vip_vip': 0},
    'L5_L23': {
        'pyr_pyr': 0.032,
        'pyr_pv': 0.075,
        'pyr_sst': 0.075,
        'pyr_vip': 0.075,
        'pv_pyr': 0.,
        'pv_pv': 0.,
        'pv_sst': 0,
        'pv_vip': 0,
        'sst_pyr': 0.,
        'sst_pv': 0.,
        'sst_sst': 0,
        'sst_vip': 0.,
        'vip_pyr': 0,
        'vip_pv': 0,
        'vip_sst': 0.,
        'vip_vip': 0},
    'L5_L4': {
        'pyr_pyr': 0.007,
        'pyr_pv': 0.003,
        'pyr_sst': 0.003,
        'pyr_vip': 0.003,
        'pv_pyr': 0.0003,
        'pv_pv': 0.,
        'pv_sst': 0,
        'pv_vip': 0,
        'sst_pyr': 0.0003,
        'sst_pv': 0.,
        'sst_sst': 0,
        'sst_vip': 0.,
        'vip_pyr': 0,
        'vip_pv': 0,
        'vip_sst': 0.,
        'vip_vip': 0},
    'L5_L6': {
        'pyr_pyr': 0.057,
        'pyr_pv': 0.028,
        'pyr_sst': 0.028,
        'pyr_vip': 0.028,
        'pv_pyr': 0.020,
        'pv_pv': 0.008,
        'pv_sst': 0,
        'pv_vip': 0,
        'sst_pyr': 0.020,
        'sst_pv': 0.008,
        'sst_sst': 0,
        'sst_vip': 0.008,
        'vip_pyr': 0,
        'vip_pv': 0,
        'vip_sst': 0.008,
        'vip_vip': 0},
    'L6_L23': {
        'pyr_pyr': 0.008,
        'pyr_pv': 0.004,
        'pyr_sst': 0.004,
        'pyr_vip': 0.004,
        'pv_pyr': 0.,
        'pv_pv': 0.,
        'pv_sst': 0,
        'pv_vip': 0,
        'sst_pyr': 0.,
        'sst_pv': 0.,
        'sst_sst': 0,
        'sst_vip': 0.,
        'vip_pyr': 0,
        'vip_pv': 0,
        'vip_sst': 0.,
        'vip_vip': 0},
    'L6_L4': {
        'pyr_pyr': 0.045,
        'pyr_pv': 0.106,
        'pyr_sst': 0.106,
        'pyr_vip': 0.106,
        'pv_pyr': 0.,
        'pv_pv': 0.,
        'pv_sst': 0,
        'pv_vip': 0,
        'sst_pyr': 0.,
        'sst_pv': 0.,
        'sst_sst': 0,
        'sst_vip': 0.,
        'vip_pyr': 0,
        'vip_pv': 0,
        'vip_sst': 0.,
        'vip_vip': 0},
    'L6_L5': {
        'pyr_pyr': 0.020,
        'pyr_pv': 0.009,
        'pyr_sst': 0.009,
        'pyr_vip': 0.009,
        'pv_pyr': 0.,
        'pv_pv': 0.,
        'pv_sst': 0,
        'pv_vip': 0,
        'sst_pyr': 0.,
        'sst_pv': 0.,
        'sst_sst': 0,
        'sst_vip': 0.,
        'vip_pyr': 0,
        'vip_pv': 0,
        'vip_sst': 0.,
        'vip_vip': 0},
    }

class ParamDict(dict):
    """ This is just a convenient class used to generate an error if the
        provided key is not present on the original dictionary. This prevents
        undeclared entries to go unnoticed so it is easier to notice when a
        non-existent connection is being handled.
    """
    def __setitem__(self, key, value):
        if key not in self:
            raise KeyError(f'You tried to change {key}, which is has not been '
                           f'previously initilized on the model. You are only '
                           f'supposed to change existent parameters')
        dict.__setitem__(self, key, value)

    def update(self, *args, **kwargs):
        for key, val in dict(*args, **kwargs).items():
            if key not in self:
                raise KeyError(f'You tried to change {key}, which is has not been '
                               f'previously initilized on the model. You are only '
                               f'supposed to change existent parameters')
            self[k] = v

class ParameterDescriptor:
    """ Parent class that contains parameters.

    Attributes:
        layer (str): Represent layer.
        model_path (str): Path to where models are stored.
        constants (dict): Constants that will be used for all elements.
        models (dict): Models of the class.

    Methods:
        change_parameters: Executes code provided that will change specific
            parameters. This gives finer control than the usual procedure,
            which filters general templates.
    """
    def __init__(self, layer, model_path):
        self.layer = layer
        # Numerator as in L/256 that will define a decay roughly equal to
        # a tau/(tau+dt)
        self.constants = {'n_bits': 4, 'max_numerator': 254}
        model_path = os.path.expanduser(model_path)
        self.model_path = os.path.join(model_path, 'orca_workspace', 'equations', "")
        self.models = {}

    def change_parameters(self, change_params):
        """ This functions will execute functions provided via change_params,
            which must receive descriptor object as a single argument, and
            must call filter_params after changes. Dictionaries must
            correspond with already existing attributes.
        """
        change_params(self)

class ConnectionDescriptor(ParameterDescriptor):
    """ This class describes the standard characterists of the connections. 
        Note that some keys are not always declared. For instance, in
        not all connections from probability dictionary are present in
        plasticity dictionary. This is just because they are not usually
        simulated. If they are to be simulated, an entry should be added.

        When adding a new/different neuronal population to this WTA building
        block, ensuing connections should be identified with a key in syn_*_prob,
        syn_*_plast, and syn_model_vals. New plasticity models can be added
        likewise. Note that an entry for the given connection in syn_sample_vars
        is optional.
        
    Attributes:
        models (dict): Synaptic models available.
        input_prob (dict): Connection probability of inputs.
        input_plast (dict): Plasticity types of inputs.
        intra_prob (dict): Connection probability of intralaminar projections.
        intra_plast (dict): Plasticity types of intralaminar projections.
        base_vals (dict): General paramaters, as defined by layer and plasticty
            types defined. Goes to set_params()
        sample (dict): Variables that will be sampled.
        reinit_var (dict): Variables used for plasticity of type 'reinit'.
    """
    def __init__(self, layer, conn_type,
                 model_path=git.Repo('.').git.rev_parse('--show-toplevel')):
        """ Initializes connection descriptor.

        Args:
            conn_type (str): String that will define which dictionary will
                be used. It can be 'input', 'intra', or 'inter'.
        """
        super().__init__(layer, model_path)
        self.models['static'] = SynapseEquationBuilder(base_unit='quantized',
            plasticity='non_plastic')
        self.models['stdp'] = SynapseEquationBuilder(base_unit='quantized',
            plasticity='quantized_stochastic_stdp')
        self.models['redsymstdp'] = SynapseEquationBuilder(base_unit='quantized',
            plasticity='quantized_stochastic_stdp',
            pairing = 'stochastic_reduced_symmetric')
        self.models['hredsymstdp'] = SynapseEquationBuilder(base_unit='quantized',
            plasticity='quantized_stochastic_stdp',
            pairing = 'stochastic_reduced_symmetric',
            compensatory_process = 'stochastic_heterosynaptic')
        self.models['adp'] = SynapseEquationBuilder.import_eq(
                self.model_path + 'StochSynAdp.py')
        self.models['altadp'] = SynapseEquationBuilder.import_eq(
                self.model_path + 'StochAdpIin.py')
        self.models['istdp'] = SynapseEquationBuilder(base_unit='quantized',
            plasticity='quantized_stochastic_istdp')
        self.models['reinit'] = SynapseEquationBuilder(base_unit='quantized',
                plasticity='quantized_stochastic_stdp',
                structural_plasticity='stochastic_counter')

        if conn_type == 'input':
            self.probabilities = ParamDict(syn_input_prob[self.layer])
            self.plasticities = ParamDict(syn_input_plast[self.layer])
        elif conn_type == 'intra':
            self.probabilities = ParamDict(syn_intra_prob[self.layer])
            self.plasticities = ParamDict(syn_intra_plast[self.layer])
        elif conn_type == 'inter':
            self.probabilities = ParamDict(syn_inter_prob[self.layer])
            self.plasticities = ParamDict(syn_inter_plast[self.layer])
        self.base_vals = {}
        self.sample = {}
        self.reinit_vars = {}

    def filter_params(self):
        """ Update parameters that will be used to build synapse model. This
            is done by changing the values of attributes base_vals, sample, 
            and reinit_vars according to what was set in probabilities and
            plasticities attributes. If the variable you want to change cannot be
            accessed throught the latter dictionaries, change them directly
            in the former after calling this method.
        """
        # Filter parameters from template according to plasticity and
        # probility of each connection
        conn_vals = {}
        for conn, plast in self.plasticities.items():
            if self.probabilities[conn] == 0:
                continue
            try:
                conn_vals[conn] = syn_model_vals[plast][conn]
            except KeyError as e:
                raise type(e)(f'{e} is not present on '
                    f'syn_model_vals. Please add it according '
                    f'to the template standard').with_traceback(
                        sys.exc_info()[2])

        for conn in conn_vals.keys():
            self.base_vals[conn] = process_base_vars(
                conn_vals[conn],
                self.constants)

        for conn in self.base_vals.keys():
            plast = self.plasticities[conn]
            try:
                self.sample[conn] = process_sample_vars(
                    syn_sample_vars[plast][conn],
                    {**self.base_vals[conn], **self.constants})
            except KeyError as e:
                raise type(e)(f'{e} is not present on '
                    f'syn_sample_vars. Please add it according '
                    f'to the template standard').with_traceback(
                        sys.exc_info()[2])

        for conn, plast in self.plasticities.items():
            # At the moment only works for reinit connections
            if plast == 'reinit':
                try:
                    self.reinit_vars[conn] = ParamDict(reinit_vars[conn])
                except KeyError as e:
                    raise type(e)(f'{e} is not present on '
                        f'reinit_vars. Please add it according '
                        f'to the template standard').with_traceback(
                            sys.exc_info()[2])

class PopulationDescriptor(ParameterDescriptor):
    """ This class describes the standard characterists of the populations.

        If you want a new/different neuronal population to be added to this
        WTA building block, an entry to neu_pop_plast, neu_pop, and neu_model_vals
        must be added. Note that population defined here will also affect how 
        connections are made. For instance, if a cell type is not declared, 
        changing parameters related to this type later on will have no actual
        effect on the model.

    Attributes:
        models (dict): Neuronal models available.
        group_prop (dict): Main values to calculate proportions of
            subgroups. At least one of each type (Pyr, PV, SST, or VIP cells)
            are necessary as Brian2 cannot create a neuron group with N=0.
        base_vals (dict): General parameters, after filtering. Goes to
            set_params()
        sample (dict): Variables that will be sampled.
        e_ratio (flot): Proportion that will scale total number of
            excitatory cells in each layer.
    """
    def __init__(self, layer,
                 model_path=git.Repo('.').git.rev_parse('--show-toplevel')):
        super().__init__(layer, model_path)
        self.models['static'] = NeuronEquationBuilder(base_unit='quantized',
            position='spatial')
        self.models['adapt'] = NeuronEquationBuilder(base_unit='quantized',
            intrinsic_excitability='threshold_adaptation',
            position='spatial')

        self.group_plast = ParamDict(neu_pop_plast)
        self.group_prop = ParamDict(neu_pop[self.layer])
        self.e_ratio = 1.0
        self.base_vals = {}
        self.sample = {}
        self.groups = {}
        
    def filter_params(self):
        """ Filter parameters that will be used to build neuron model. This
            is done by changing the values of attributes base_vals, sample, and
            groups according to what was set in group_plast, group_prop, and
            e_ratio attributes. If the variable you want to change cannot be
            accessed throught the latter dictionaries, change them directly
            in the former after calling this method.
        """
        # Filter parameters from template according to plasticity for
        # each population
        group_vals = {}
        for conn, plast in self.group_plast.items():
            try:
                group_vals[conn] = neu_model_vals[plast][conn]
            except KeyError as e:
                raise type(e)(f'{e} is not present on '
                    f'neu_model_vals. Please add it according '
                    f'to the template standard').with_traceback(
                        sys.exc_info()[2])

        temp_pop = {}
        # Adjust excitatory populatin if necessary
        self.group_prop['n_exc'] = int(self.e_ratio * self.group_prop['n_exc'])
        num_inh = int(self.group_prop['n_exc']/self.group_prop['ei_ratio'])
        temp_pop['pyr_cells'] = {'num_neu': self.group_prop['n_exc']}
        for inh_pop, ratio in self.group_prop['inh_ratio'].items():
            temp_pop[inh_pop] = {'num_neu': int(num_inh * ratio)}

        for pop, n_inp in self.group_prop['num_inputs'].items():
            temp_pop[pop].update({'num_inputs': n_inp})

        self.groups = ParamDict(temp_pop)

        for conn, plast in self.group_plast.items():
            self.base_vals[conn] = process_base_vars(
                group_vals[conn],
                self.constants)

        for neu_group in self.group_plast.keys():
            self.sample[neu_group] = process_sample_vars(
                neu_sample_vars[neu_group],
                {**self.base_vals[neu_group], **self.constants})

def process_base_vars(base_objects, reference_vals=None):
    ''' This function filter the necessary parameters from provided
        reference dictionaries to determine base values.

    Args:
        base_objects (dict of list): Contains keys and values
            identifying the parameters. If it contains a
            lambda function, its argument name must be present
            as a key in reference_vals.
        reference_vals (dict): Base parameters used on base
            values. If None, the provided base_objects is used as a 
            reference.
    Returns:
        processed_objects (list): Contains parameters that will be used
            as base paramaters.
    '''
    processed_objects = copy.deepcopy(base_objects)
    if not reference_vals:
        reference_vals = copy.deepcopy(base_objects)

    for var in base_objects:
        if callable(base_objects[var]):
            processed_objects[var] = process_dynamic_values(
                base_objects[var], reference_vals)

    return ParamDict(processed_objects)

def process_sample_vars(sample_objects, reference_vals):
    ''' This function filter the necessary parameters from provided
        reference dictionaries to determine sampling process.

    Args:
        sample_objects (dict of list): Contains keys and values
            identifying the samplying process. If it contains a
            lambda function, its argument name must be present
            as a key in reference_vals.
        reference_vals (dict): Base parameters used on sampling
            process.
    Returns:
        sample_list (list of dict): Contains parameters that will be used
            by sampling function.
    '''
    sample_list = []
    for var_name, var_settings in sample_objects.items():
        # Get values from base val if they are not numbers
        variable_mean = reference_vals[var_name]
        unit = var_settings['unit']
        sign = var_settings['sign']
        if callable(sign):
            sign = process_dynamic_values(sign, reference_vals)
        unit *= sign
        variable_mean /= np.abs(unit)
        clip_min = var_settings['min']
        if callable(clip_min):
            clip_min = process_dynamic_values(clip_min, reference_vals)
        clip_max = var_settings['max']
        if callable(clip_max):
            clip_max = process_dynamic_values(clip_max, reference_vals)

        # Note that variable_mean is not actually used for uniform distributions
        sample_list.append(ParamDict({'variable': var_name,
                                      'dist_param': np.abs(variable_mean),
                                      'dist_type': var_settings['dist_type'],
                                      'unit': unit,
                                      'clip_min': clip_min,
                                      'clip_max': clip_max}))

    return sample_list

def process_dynamic_values(lambda_func, reference_dict):
    """ Evaluates parameters defined as lambda functions.

    Args:
        lambda_func (callable): Function to be evaluated.
        reference_dict (dict): Contains function's argument
            as a key.
    """
    var_name = lambda_func.__code__.co_varnames[0]
    return lambda_func(reference_dict[var_name])
