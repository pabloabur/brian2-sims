from brian2 import defaultclock, prefs, ms

from teili import TeiliNetwork
from orca_wta import ORCA_WTA
from orca_column import orcaColumn

import numpy as np

def change_params_conn1(desc):
    # Changes some intralaminar connections
    desc.plasticities['sst_pv'] = 'static'
    desc.plasticities['pyr_pyr'] = 'hredsymstdp'
    desc.probabilities['pyr_pyr'] = 1.0
    desc.filter_params()
    # TODO try normal resolution here, just delete w_max
    desc.base_vals['pyr_pyr']['w_max'] = 15
    desc.base_vals['pv_pyr']['w_max'] = 15
    desc.base_vals['sst_pyr']['w_max'] = 15
    desc.base_vals['pyr_pyr']['stdp_thres'] = 15

def change_params_pop1(desc):
    # Changes proportions of the EI population
    desc.e_ratio = .00224
    desc.filter_params()
    # TODO delete it and use normal resolution
    for pop in desc.base_vals:
        desc.base_vals[pop]['I_min'] = -256*mA
        desc.base_vals[pop]['I_max'] = 256*mA

def change_params_pop2(desc):
    # Changes proportions of the EI population
    desc.e_ratio = .01
    desc.group_prop['ei_ratio'] = 4
    desc.group_prop['inh_ratio']['pv_cells'] = .68
    desc.group_prop['inh_ratio']['sst_cells'] = .20
    desc.group_prop['inh_ratio']['vip_cells'] = .12
    desc.filter_params()
    # TODO delete it and use normal resolution
    for pop in desc.base_vals:
        desc.base_vals[pop]['I_min'] = -256*mA
        desc.base_vals[pop]['I_max'] = 256*mA

def change_params_conn4(desc):
    # Changes interlaminar parameters
    desc.plasticities['pyr_pyr'] = 'redsymstdp'
    desc.probabilities['pyr_pyr'] = .3
    desc.filter_params()
    desc.base_vals['pyr_pyr']['w_max'] = 15
    desc.base_vals['pyr_pyr']['stdp_thres'] = 7

def change_params_conn2(desc):
    # Changes input parameters
    desc.plasticities['ff_pyr'] = 'redsymstdp'
    desc.filter_params()
    desc.base_vals['ff_pyr']['w_max'] = 15
    desc.base_vals['ff_pyr']['stdp_thres'] = 7

def change_params_conn3(desc):
    # Changes input parameters
    desc.plasticities['ff_pyr'] = 'redsymstdp'
    desc.probabilities['ff_pyr'] = 0.7
    desc.probabilities['ff_pv'] = 1
    desc.probabilities['ff_sst'] = 1
    desc.filter_params()
    desc.base_vals['ff_pyr']['w_max'] = 15
    desc.base_vals['ff_pyr']['stdp_thres'] = 7

defaultclock.dt = 1*ms
prefs.codegen.target = "numpy"

cmc = orcaColumn(['L4', 'L5'])

conn_modifier = {'L4': change_params_conn1, 'L5': change_params_conn1}
pop_modifier = {'L4': change_params_pop1, 'L5': change_params_pop2}
column.create_layers(pop_modifier, conn_modifier)
conn_modifier = {'L4_L5': change_params_conn4}
column.connect_layers(conn_modifier)
conn_modifier = {'L4': change_params_conn2, 'L5': change_params_conn3}
column.connect_inputs(relay_cells, 'ff', conn_modifier)

# Number of thalamic projections
num_thal = 902
# Number of neurons accumulated
neurons_accum = [0]
neurons_accum.extend(np.cumsum(num_layer))
# External populations related to background activity
#                            2/3e  2/3i  4e    4i    5e    5i    6e    6i
background_layer = np.array([1600, 1500 ,2100, 1900, 2000, 1900, 2900, 2100])
# Prob. connection table: from colum to row
#                 L2/3e 	L2/3i 	L4e 	L4i   L5e   L5i 	L6e    L6i 	   Th
p_conn = np.array([[     ,  0.169,      ,      ,      ,   ,          ,   ,           ], #L2/3e
                   [     ,  0.137,      ,      ,      ,   ,          ,   ,           ], #L2/3i
                   [     ,       ,      , 0.135,      ,       ,      ,   ,     0.0983], #L4e
                   [     ,       ,      , 0.160,      ,   ,          ,   ,     0.0619], #L4i
                   [     ,       ,      ,      ,      , 0.373,       ,   ,           ], #L5e
                   [     ,       ,      ,      ,      , 0.316,       ,   ,           ], #L5i
                   [     ,       ,      ,      ,      ,      ,       , 0.225,  0.0512], #L6e
                   [     ,       ,      ,      ,      ,      ,       , 0.144,  0.0196]]) #L6i
# TODO set
# Use defaults of runParamsParellel(). Protocol=0; Fig.6; g=4,
# bg_type=0 (layer specific), bg_freq=8.0, stim=0 (bg noise)
#  N.B. nsyn_type=0, but I am not using nsyn, just probabilities

# Initializations
# TODO exc weight, exc/inh delay, membrane potential from gaussian?
#   vm='-58.0*mV + 10.0*mV*randn()'
#   thal_con[r].w = 'clip((w_thal + std_w_thal*randn()),w_thal*0.0, w_thal*inf)'
#       std_w_thal = w_thal*0.1
#       w_thal = w_ex*pA
#   mem_tau   = 10.0*ms
#   refrac_tau = 2*ms
#   std_w_ex = 0.1*w_ex
# TODO equivalent of below
d_ex = 1.5*ms      	# Excitatory delay
std_d_ex = 0.75*ms 	# Std. Excitatory delay
d_in = 0.80*ms      # Inhibitory delay
std_d_in = 0.4*ms  	# Std. Inhibitory delay
tau_syn = 0.5*ms    # Post-synaptic current time constant
w_ex = 87.8*pA		   	# excitatory synaptic weight
std_w_ex = 0.1*w_ex     # standard deviation weigth
