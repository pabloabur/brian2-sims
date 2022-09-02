#trials = 10
#last_sequence_t = training_duration - testing_duration - int(
#    sequence_duration/defaultclock.dt)*defaultclock.dt*trials
#neu_rates = neuron_rate(spikemon_exc_neurons, kernel_len=10*ms,
#    kernel_var=1*ms, simulation_dt=defaultclock.dt,
#    interval=[last_sequence_t, training_duration - testing_duration], smooth=True,
#    trials=trials)
#
#conn_pattern = label_ensembles(sequence, neu_rates, rate_thr=50*Hz)
#conn_desc = ConnectionDescriptor('L4', 'intra')
#conn_desc.plasticities['pyr_pyr'] = 'static'
#conn_desc.filter_params()
#conn_desc.base_vals['pyr_pyr']['delay'] = 0*ms
#pyr_readout = Connections(
#    column.col_groups['L4'].groups['pyr_cells'], readout,
#    equation_builder=conn_desc.models['static'](),
#    method=stochastic_decay,
#    name='pyr_readout'
#    )
#try:
#    pyr_readout.connect(i=conn_pattern['source'], j=conn_pattern['target'])
#except:
#    pyr_readout.connect()
#pyr_readout.set_params(conn_desc.base_vals['pyr_pyr'])
#pyr_readout.weight = 1
#Net.add(pyr_readout)
