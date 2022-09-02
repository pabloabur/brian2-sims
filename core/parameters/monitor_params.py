from brian2 import ms
import numpy as np

rng = np.random.default_rng(12345)
selected_cells = rng.choice(range(49), 5, replace=False)

monitor_params = {'spikemon_pyr_cells': {'group': 'pyr_cells'},
                  'spikemon_pv_cells': {'group': 'pv_cells'},
                  'spikemon_sst_cells': {'group': 'sst_cells'},
                  'spikemon_vip_cells': {'group': 'vip_cells'},
                  'statemon_pyr_cells': {'group': 'pyr_cells',
                                         'variables': ['I'],
                                         'record': selected_cells,
                                         'mon_dt': 1*ms},
                  'statemon_conn_ff_pyr': {'group': 'ff_pyr',
                                           'variables': ['w_plast'],
                                           'record': True,
                                           'mon_dt': 500*ms},
                  'statemon_static_conn_ff_pyr': {'group': 'ff_pyr',
                                                  'variables': ['weight'],
                                                  'record': True,
                                                  'mon_dt': 143996*ms},
                  'statemon_conn_pyr_pyr': {'group': 'pyr_pyr',
                                            'variables': ['w_plast'],
                                            'record': True,
                                            'mon_dt': 500*ms},
                  'rate_pyr_cells': {'group': 'pyr_cells'},
                  'rate_pv_cells': {'group': 'pv_cells'}
                  }
