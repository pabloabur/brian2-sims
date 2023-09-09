suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(dplyr))

data_path<-'sim_data/stdp_test/'

df_events <- read.csv(file.path(data_path, 'events_spikes.csv'))
df_weights <- read.csv(file.path(data_path, 'synapse_vars.csv'))
df_vars <- read.csv(file.path(data_path, 'state_vars.csv'))

# TODO ca pre and ca post (for one pair only), with events overlay
# for ca just pick one e.g. id 0 to 0, but no ref_
# statemon_pre_neurons and statemon_post_neurons
ca_trace <- df_variables %>%
           filter(variable=='Ca') %>%
           ggplot(aes(x=time_ms, y=value)) +
           geom_line()

# TODO better show weight evolution profile
w_trace <- df_weights %>%
           filter(variable=='w_plast' & monitor=='stdp_in_w') %>%
           ggplot(aes(x=time_ms, y=value, group=id, color=id)) +
           geom_line()
