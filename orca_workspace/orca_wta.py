""" This module can be used to scale-up models published by Wang et al. (2018).
    """

import pickle
import numpy as np

from brian2 import ms, amp, mA, Hz, mV, ohm, ExplicitStateUpdater,\
        PoissonGroup, PoissonInput, SpikeMonitor, StateMonitor, PopulationRateMonitor

from teili.building_blocks.building_block import BuildingBlock
from teili.core.groups import Neurons, Connections
from teili.tools.group_tools import add_group_activity_proxy,\
    add_group_params_re_init, add_group_param_init

from teili.tools.misc import DEFAULT_FUNCTIONS
from SLIF_run_regs import add_alt_activity_proxy
from utils.SLIF_utils import expand_state_variables
from parameters.monitor_params import monitor_params

stochastic_decay = ExplicitStateUpdater('''x_new = f(x,t)''')

class orcaWTA(BuildingBlock):
    """A WTA with diverse inhibitory population. This could represent a single
       layer in a cortical sheet.

    Attributes:
        _groups (dict): Contains all synapses and neurons of the building
            block. For convenience, keys identifying a neuronal population 'x'
            should be 'x_cells', whereas keys identifying a synapse between
            population 'x' and 'y' should be 'x_y'. At the moment, options
            available are 'pyr', 'pv', 'sst', and 'vip'.
        layer (str): Indicates which cortical layer the building block
            represents
    """

    def __init__(self,
                 layer,
                 name,
                 conn_params,
                 pop_params,
                 verbose=False,
                 monitor=False,
                 noise=False):
        """ Generates building block with specified characteristics and
                elements described by Wang et al. (2018).

        Args:
            num_exc_neurons (int): Size of excitatory population.
            ei_ratio (int): Ratio of excitatory versus inhibitory population,
                that is ei_ratio:1 representing exc:inh
            layer (str): Indicates cortical layer that is supposed
                to be mimicked by WTA network. It can be L23, L4, L5,
                or L6.
            name (str, required): Name of the building_block population
            conn_params (ConnectionDescriptor): Class which holds building_block
                specific parameters. It can be find on tutorials/orca_params.py
            pop_params (PopulationDescriptor): Class which holds building_block
                specific parameters. It can be find on tutorials/orca_params.py
            verbose (bool, optional): Flag to gain additional information
            monitor (bool, optional): Flag to indicate presence of monitors
            noise (bool, optional): Flag to determine if background noise is to
                be added to neurons. This is generated with a poisson process.
        """
        BuildingBlock.__init__(self,
                               name,
                               None,
                               None,
                               None,
                               verbose,
                               monitor)

        self._groups = {}
        self.layer = layer

        add_populations(self._groups,
                        group_name=self.name,
                        pop_params=pop_params,
                        verbose=verbose,
                        noise=noise
                        )
        add_connections([*conn_params.base_vals],
                        self._groups,
                        group_name=self.name,
                        conn_params=conn_params,
                        verbose=verbose)

    def add_input(self,
                  input_group,
                  source_name,
                  targets,
                  conn_params,
                  extra_name=''):
        """ This functions add an input group and connections to the building
            block.

        Args:
            input_group (brian2.NeuronGroup): Input to building block.
            source_name (str): Name of the input group to be registered. These
                must be the names of neuron groups as identified in
                dictionaries containing parameters.
            targets (list of str): Name of the postsynaptic groups as
                stored in _groups.
            conn_params (ConnectionDescriptor): Class which holds building_block
                specific parameters. It can be find on tutorials/orca_params.py
            extra_name (str): Aditional name that is appened to source_name
                provided. This is useful when scaling up models as groups
                with repeated names in Brian2 will raise an error.
        """
        conn_ids = [source_name + '_' + target.split('_')[0] for target in targets]
        self.input_groups[extra_name+self.name+source_name+'_cells'] = input_group
        add_connections(conn_ids,
                        self._groups,
                        group_name=extra_name+self.name,
                        conn_params=conn_params,
                        input_groups=self.input_groups)

    def create_monitors(self, monitor_params):
        """ This creates monitors according to dictionary provided.
        
        Args:
            monitor_params (dict): Contains information about monitors.
                For convenience, some words MUST be present in the name of the
                monitored group, according to what the user wants to record.
                At the moment, for tutorials/orca_wta.py, these are:
                'spike', 'state' (with 'cells' or 'conn'), and 'rate'.
                This must be followed by 'x_cell' or 'conn_x_y' accordingly
                (check class docstring for naming description).
        """
        for key, val in monitor_params.items():
            if 'spike' in key:
                self.monitors[key] = SpikeMonitor(self._groups[val['group']],
                                                  name=self.name + key)
            elif 'state' in key:
                if 'cells' in key:
                    self.monitors[key] = StateMonitor(self._groups[val['group']],
                                                      variables=val['variables'],
                                                      record=val['record'],
                                                      dt=val['mon_dt'],
                                                      name=self.name + key)
                elif 'conn' in key:
                    try:
                        self.monitors[key] = StateMonitor(self._groups[val['group']],
                                                          variables=val['variables'],
                                                          record=val['record'],
                                                          dt=val['mon_dt'],
                                                          name=self.name + key)
                    except KeyError:
                        self.monitors[key] = StateMonitor(self.input_groups[val['group']],
                                                          variables=val['variables'],
                                                          record=val['record'],
                                                          dt=val['mon_dt'],
                                                          name=self.name + key)
            elif 'rate' in key:
                self.monitors[key] = PopulationRateMonitor(self._groups[val['group']],
                                                           name=self.name + key)

    def save_data(self, monitor_params, path, block=0):
        """ Saves monitor data to disk.
        
        Args:
            monitor_params (dict): Contains information about monitors.
                For convenience, some words MUST be present in the key of the
                monitored group, according to what the user wants to record.
                At the moment, for tutorials/orca_wta.py, these are:
                'spike', 'state' (with 'cells' or 'conn'), and 'rate'.
                This must be followed by 'x_cell' or 'conn_x_y' accordingly
                (check class docstring for naming description).
            block (int, optional): If saving in batches, save with batch number.
        """
        selected_keys = [x for x in monitor_params.keys() if 'spike' in x]
        # TODO np.savez(outfile, **{x_name: x, y_name: y}
        pv_indices, sst_indices, vip_indices = np.array([]), np.array([]), np.array([])
        pv_times, sst_times, vip_times = np.array([]), np.array([]), np.array([])
        for key in selected_keys:
            # Concatenate data from inhibitory population
            if 'pv_cells' in key:
                pv_times = np.array(self.monitors[key].t/ms)
                pv_indices = np.array(self.monitors[key].i)
            elif 'sst_cells' in key:
                sst_times = np.array(self.monitors[key].t/ms)
                sst_indices = np.array(self.monitors[key].i)
                sst_indices += self._groups['pv_cells'].N
            elif 'vip_cells' in key:
                vip_times = np.array(self.monitors[key].t/ms)
                vip_indices = np.array(self.monitors[key].i)
                vip_indices += (self._groups['pv_cells'].N + self._groups['sst_cells'].N)
            elif 'pyr_cells' in key:
                pyr_times = np.array(self.monitors[key].t/ms)
                pyr_indices = np.array(self.monitors[key].i)

        inh_spikes_t = np.concatenate((pv_times, sst_times, vip_times))
        inh_spikes_i = np.concatenate((pv_indices, sst_indices, vip_indices))
        sorting_index = np.argsort(inh_spikes_t)
        inh_spikes_t = inh_spikes_t[sorting_index]
        inh_spikes_i = inh_spikes_i[sorting_index]

        np.savez(path + f'rasters_{block}.npz',
                 exc_spikes_t=pyr_times,
                 exc_spikes_i=pyr_indices,
                 inh_spikes_t=inh_spikes_t,
                 inh_spikes_i=inh_spikes_i
                 )

        # If there are only a few samples, smoothing operation can create an array
        # which is incompatible with array with spike times. This is then addressed
        # before saving to disk
        selected_keys = [x for x in monitor_params.keys() if 'rate' in x]
        for key in selected_keys:
            if 'pyr_cells' in key:
                exc_rate_t = np.array(self.monitors[key].t/ms)
                if self.monitors[key].rate:
                    exc_rate = np.array(self.monitors[key].smooth_rate(width=10*ms)/Hz)
                else:
                    exc_rate = np.array(self.monitors[key].rate/Hz)
            if 'pv_cells' in key:
                inh_rate_t = np.array(self.monitors[key].t/ms)
                if self.monitors[key].rate:
                    inh_rate = np.array(self.monitors[key].smooth_rate(width=10*ms)/Hz)
                else:
                    inh_rate = np.array(self.monitors[key].rate/Hz)
        # Sometimes the smoothed rate calculated on last block is not the
        # same size as time array. In this cases, raw rate is considered. This
        # means artifacts at the end of simulation
        if len(exc_rate_t) != len(exc_rate):
            exc_rate = np.array(self.monitors['rate_pyr_cells'].rate/Hz)
            inh_rate = np.array(self.monitors['rate_pv_cells'].rate/Hz)
        selected_keys = [x for x in monitor_params.keys() if 'state' in x]
        for key in selected_keys:
            if 'pyr_cells' in key:
                I = self.monitors[key].I
        np.savez(path + f'traces_{block}.npz',
                 I=I, exc_rate_t=exc_rate_t, exc_rate=exc_rate,
                 inh_rate_t=inh_rate_t, inh_rate=inh_rate,
                 )

        # Save targets of recurrent connections as python object
        recurrent_ids, ff_ids, ffi_ids = [], [], []
        for row in range(self._groups['pyr_cells'].N):
            recurrent_ids.append(list(self._groups['pyr_pyr'].j[row, :]))
        for row in range(self.input_groups['L4_ff_cells'].N):
            ff_ids.append(list(self.input_groups['L4_ff_pyr'].j[row, :]))
        for row in range(self._groups['pv_cells'].N):
            ffi_ids.append(list(self.input_groups['L4_ff_pv'].j[row, :]))
        #selected_keys = [x for x in monitor_params.keys() if 'state' in x]
        #for key in selected_keys:
        ff_pyr_w = np.full((self.input_groups['L4_ff_cells'].N,
                            self.groups['pyr_cells'].N),
                           np.nan)
        i_indices = self.input_groups['L4_ff_pyr'].i[:]
        j_indices = self.input_groups['L4_ff_pyr'].j[:]
        ff_pyr_w[i_indices, j_indices] = self.input_groups['L4_ff_pyr'].w_plast

        pyr_pyr_w = np.full((self.groups['pyr_cells'].N,
                             self.groups['pyr_cells'].N),
                            np.nan)
        i_indices = self.groups['pyr_pyr'].i[:]
        j_indices = self.groups['pyr_pyr'].j[:]
        pyr_pyr_w[i_indices, j_indices] = self.groups['pyr_pyr'].w_plast

        np.savez_compressed(path + f'matrices_{block}.npz',
            rf=self.monitors['statemon_conn_ff_pyr'].w_plast.astype(np.uint8),
            rfw=self.monitors['statemon_static_conn_ff_pyr'].weight.astype(np.uint8),
            recw=self.monitors['statemon_conn_pyr_pyr'].w_plast.astype(np.uint8),
            rec_ids=recurrent_ids, ff_ids=ff_ids, ffi_ids=ffi_ids,
            ff_pyr_w=ff_pyr_w,
            pyr_pyr_w=pyr_pyr_w
            )

def add_populations(_groups,
                    group_name,
                    pop_params,
                    verbose,
                    noise):
    """ This functions add populations of the building block.

    Args:
        groups (dict): Keys to all neuron and synapse groups.
        group_name (str, required): Name of the building_block population
        pop_params (dict): Parameters used to build populations.
        verbose (bool, optional): Flag to gain additional information
        noise (bool, optional): Flag to determine if background noise is to
            be added to neurons. This is generated with a poisson process.
    """
    temp_groups = {}
    for pop_id, params in pop_params.groups.items():
        neu_type = pop_params.group_plast[pop_id]
        if not params['num_neu']:
            continue
        temp_groups[pop_id] = Neurons(
            params['num_neu'],
            equation_builder=pop_params.models[neu_type](num_inputs=params['num_inputs']),
            method=stochastic_decay,
            name=group_name+pop_id,
            verbose=verbose)
        temp_groups[pop_id].set_params(pop_params.base_vals[pop_id])
    
    if noise:
        pyr_noise_cells = PoissonInput(pyr_cells, 'Vm_noise', 1, 3*Hz, 12*mV)
        pv_noise_cells = PoissonInput(pv_cells, 'Vm_noise', 1, 2*Hz, 12*mV)
        sst_noise_cells = PoissonInput(sst_cells, 'Vm_noise', 1, 2*Hz, 12*mV)
        vip_noise_cells = PoissonInput(vip_cells, 'Vm_noise', 1, 2*Hz, 12*mV)
        temp_groups.update({'pyr_noise_cells': pyr_noise_cells,
                            'pv_noise_cells': pv_noise_cells,
                            'sst_noise_cells': sst_noise_cells,
                            'vip_noise_cells': vip_noise_cells})

    _groups.update(temp_groups)

def add_connections(connection_ids,
                    _groups,
                    group_name,
                    conn_params,
                    input_groups=None,
                    verbose=False):
    """ This function adds the connections of the building block.

    Args:
        connection_ids (list of str): Identification of each connection
            to be made.
        _groups (dict): Keys to all neuron and synapse groups.
        group_name (str, required): Name of the building_block population
        conn_params (ConnectionDescriptor): Class which holds building_block
            specific parameters. It can be find on tutorials/orca_params.py
        input_groups (brian2.NeuronGroup): Contains input group in case it
            comes from outside target group.
        verbose (bool, optional): Flag to gain additional information
    """
    syn_types = conn_params.plasticities
    connectivities = conn_params.probabilities
    source_group = {}
    if input_groups:
        # Only one external input connected at a time, so is done once
        temp_name = connection_ids[0].split('_')[0] + '_cells'
        source_group[temp_name] = input_groups[group_name+temp_name]
    else:
        source_group = _groups

    for conn_id in connection_ids:
        source, target = conn_id.split('_')[0], conn_id.split('_')[1]
        connectivity = connectivities[conn_id]
        if (not source+'_cells' in source_group) or (not target+'_cells' in _groups):
            continue
        if not conn_params.probabilities[conn_id]:
            continue

        syn_type = syn_types[conn_id]
        if syn_type == 'reinit':
            connectivity = 1

        conn_mat = np.random.choice([0, 1],
                                    size=(source_group[source+'_cells'].N,
                                          _groups[target+'_cells'].N),
                                    p=[1-connectivity, connectivity])
        sources, targets = conn_mat.nonzero()
        if not np.any(sources):
            continue

        # Checks for tags on equations that require changes. In
        # my case this sets proper sum_w name of post synaptic group
        # when heterosynaptic plasticity is used
        conn_params.models[syn_type] = expand_state_variables(
            conn_params.models[syn_type], 'sumw', source)

        temp_conns = Connections(
            source_group[source+'_cells'], _groups[target+'_cells'],
            equation_builder=conn_params.models[syn_type](),
            method=stochastic_decay,
            name=group_name+conn_id
            )

        temp_conns.connect(i=sources, j=targets)
        temp_conns.set_params(conn_params.base_vals[conn_id])

        sample_vars = conn_params.sample[conn_id]
        for sample_var in sample_vars: 
            add_group_param_init([temp_conns],
                                 variable=sample_var['variable'],
                                 dist_param=sample_var['dist_param'],
                                 scale=1,
                                 unit=sample_var['unit'],
                                 distribution=sample_var['dist_type'],
                                 clip_min=sample_var['clip_min'],
                                 clip_max=sample_var['clip_max'],
                                 sizes=[len(sources)],
                                 casting_type=int)

        if syn_type == 'adp':
            dummy_unit = 1*mV
            _groups[target+'_cells'].variables.add_array(
                'activity_proxy',
                size=_groups[target+'_cells'].N,
                dimensions=dummy_unit.dim)
            _groups[target+'_cells'].variables.add_array(
                'normalized_activity_proxy',
                size=_groups[target+'_cells'].N)
            activity_proxy_group = [_groups[target+'_cells']]
            add_group_activity_proxy(activity_proxy_group,
                                     buffer_size=400,
                                     decay=150)
            temp_conns.variance_th = np.random.uniform(
                low=temp_conns.variance_th - 0.1,
                high=temp_conns.variance_th + 0.1,
                size=len(temp_conns))

        if syn_type == 'altadp':
            dummy_unit = 1*amp
            _groups[target+'_cells'].variables.add_array('activity_proxy',
                                          size=_groups[target+'_cells'].N,
                                          dimensions=dummy_unit.dim)
            _groups[target+'_cells'].variables.add_array('normalized_activity_proxy',
                                          size=_groups[target+'_cells'].N)
            add_alt_activity_proxy([_groups[target+'_cells']],
                                   buffer_size=400,
                                   decay=150)

        # If you want to have sparsity without structural plasticity,
        # set the desired connection probability only
        if syn_type == 'reinit':
            for neu in range(_groups[target+'_cells'].N):
                ffe_zero_w = np.random.choice(
                    source_group[source+'_cells'].N,
                    int(source_group[source+'_cells'].N * conn_params.probabilities[conn_id]),
                    replace=False)
                temp_conns.weight[ffe_zero_w, neu] = 0
                temp_conns.w_plast[ffe_zero_w, neu] = 0

            # TODO for reinit_var in conn_params.reinit_vars[conn_id]
            add_group_params_re_init(groups=[temp_conns],
                                     variable='w_plast',
                                     re_init_variable='re_init_counter',
                                     re_init_threshold=1,
                                     re_init_dt=conn_params.reinit_vars[conn_id]['re_init_dt'],
                                     dist_param=3,
                                     scale=1,
                                     distribution='gamma',
                                     clip_min=0,
                                     clip_max=15,
                                     variable_type='int',
                                     reference='synapse_counter')
            add_group_params_re_init(groups=[temp_conns],
                                     variable='weight',
                                     re_init_variable='re_init_counter',
                                     re_init_threshold=1,
                                     re_init_dt=conn_params.reinit_vars[conn_id]['re_init_dt'],
                                     distribution='deterministic',
                                     const_value=1,
                                     reference='synapse_counter')
            # TODO error when usinng below. Ditch tausyn reinit?
            #add_group_params_re_init(groups=[temp_conns],
            #                         variable='tausyn',
            #                         re_init_variable='re_init_counter',
            #                         re_init_threshold=1,
            #                         re_init_dt=conn_params.reinit_vars[conn_id]['re_init_dt'],
            #                         dist_param=5.5,
            #                         scale=1,
            #                         distribution='normal',
            #                         clip_min=4,
            #                         clip_max=7,
            #                         variable_type='int',
            #                         unit='ms',
            #                         reference='synapse_counter')

        if input_groups:
            input_groups[group_name+conn_id] = temp_conns
        else:
            _groups[conn_id] = temp_conns
