import itertools

from brian2 import mA

from orca_wta import orcaWTA
from parameters.orca_params import ConnectionDescriptor, PopulationDescriptor

class orcaColumn:
    """ This class defines how layer are connected within a column.

    Attributes:
        layers (list of str): Layers of the column. Note that we can have 
            abstract columns with fewer layers.
        col_groups (dict of list): Brian2 col_groups for each layer.
    """
    def __init__(self, layers):
        self.layers = layers
        self.col_groups = {x: None for x in self.layers}

    def create_layers(self, pop_modifier={}, conn_modifier={}):
        """ This method creates layers as provided on layers attribute.

        Args:
            pop_modifier (dict): Contains a function will change
                parameters of populations. Keys must be
                contained in layers.
            conn_modifier (dict): Contains a function will change
                parameters of connections. Keys must be
                contained in layers.
        """
        # Fills undeclared keys with None for convenience
        _ = [pop_modifier.setdefault(x) for x in self.layers]
        _ = [conn_modifier.setdefault(x) for x in self.layers]

        for layer in self.layers:
            conn_desc = ConnectionDescriptor(layer, 'intra')
            if conn_modifier[layer]:
                conn_desc.change_parameters(conn_modifier[layer])
            else:
                conn_desc.filter_params()

            pop_desc = PopulationDescriptor(layer)
            if pop_modifier[layer]:
                pop_desc.change_parameters(pop_modifier[layer])
            else:
                pop_desc.filter_params()

            orca = orcaWTA(layer=layer,
                           name=layer+'_',
                           conn_params=conn_desc,
                           pop_params=pop_desc,
                           monitor=True)
            self.col_groups[layer] = orca

    def connect_layers(self, conn_modifier={}):
        """ This method connects layers according to dictionary with
            connectivities.

        Args:
            conn_modifier (dict): Contains a function will change
                parameters of connections. Keys must be
                contained in pair of layers being connected.
        """
        if len(self.layers)==1:
            pairs = []
        else:
            pairs = list(itertools.permutations(self.layers))

        for pair in pairs:
            laminar_source, laminar_target = pair[0], pair[1]
            conn_name = laminar_source+'_'+laminar_target

            # Fills undeclared keys with None for convenience
            conn_modifier.setdefault(conn_name)

            conn_desc = ConnectionDescriptor(conn_name, 'inter')
            if conn_modifier[conn_name]:
                conn_desc.change_parameters(conn_modifier[conn_name])
            else:
                conn_desc.filter_params()

            source_cells = [x
                for x in self.col_groups[laminar_source]._groups.keys()
                if 'cells' in x]
            target_cells = [x
                for x in self.col_groups[laminar_target]._groups.keys()
                if 'cells' in x]
            for cell in source_cells:
                self.col_groups[laminar_target].add_input(
                    self.col_groups[laminar_source]._groups[cell],
                    cell.split('_')[0],
                    target_cells,
                    conn_desc,
                    extra_name=laminar_source+'_')

    def connect_inputs(self, input_group, input_name, conn_modifier={}):
        """ This method connects inputs to layers.

        Args:
            input_group (Brian2 group): Input group.
            input_name (str): Identication of the input.
        """
        # Fills undeclared keys with None for convenience
        _ = [conn_modifier.setdefault(x) for x in self.layers]

        for layer in self.layers:
            conn_desc = ConnectionDescriptor(layer, 'input')
            if conn_modifier[layer]:
                conn_desc.change_parameters(conn_modifier[layer])
            else:
                conn_desc.filter_params()
            self.col_groups[layer].add_input(
                input_group,
                input_name,
                ['pyr_cells', 'pv_cells', 'sst_cells', 'vip_cells'],
                conn_desc)
