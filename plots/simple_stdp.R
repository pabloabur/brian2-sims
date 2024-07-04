suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(patchwork))
suppressPackageStartupMessages(library(latex2exp))
suppressPackageStartupMessages(library(stringr))

library(wesanderson)
color_map <- wes_palette('GrandBudapest1')

library(argparser)
include('plots/parse_inputs.R')

df_weights <- read.csv(file.path(argv$source, 'synapse_vars.csv'))
df_vars <- read.csv(file.path(argv$source, 'state_vars.csv'))
# N.B. Only presynaptic neurons were recorded here
df_events <- read.csv(file.path(argv$source, 'events_spikes.csv'))

df_pre_events <- df_events %>% filter(id==0) %>% filter(time_ms>690 & time_ms<890)
df_pre_events$id <- df_pre_events$id + 1.1
df_pre_ca_trace <- df_vars %>%
    filter(id==0) %>%
    filter(time_ms>690 & time_ms<890) %>%
    filter(variable=='Ca' & monitor=='statemon_pre_neurons')
# Create an histogram to place on top of pre_ca_trace line
ca_traces <- ggplot() +
    geom_line(data=df_pre_ca_trace, aes(x=time_ms, y=value)) +
    geom_histogram(data=df_pre_events, aes(x=time_ms), fill=color_map[2],
                   alpha=0.3, binwidth=1) +
    labs(x=element_blank(), y=TeX(r'($x_i$ (a.u))')) +
    theme_bw() + theme(panel.grid.minor=element_blank(), text = element_text(size=16))

df_post_ca_trace <- df_vars %>%
    filter(id==0) %>%
    filter(variable=='Ca' & monitor=='statemon_post_neurons') %>%
    filter(time_ms>690 & time_ms<890) %>%
    ggplot(aes(x=time_ms, y=value)) +
    labs(x='time (ms)', y=TeX(r'($x_j$ (a.u))')) +
    theme_bw() + theme(panel.grid.minor=element_blank(), text = element_text(size=16)) +
    geom_line()

b_rang <- c(0, 310, 420, 720, 800, 1110, 1210, 1510, 1610, 2010, 2080, 2320)
w_trace <- df_weights %>%
    filter(variable=='w_plast') %>%
    mutate(monitor = str_replace_all(monitor, c('ref_statemon_post_synapse' = 'Original',
                                                'statemon_post_synapse' = 'Proposed'))) %>%
    mutate(value=value * 1000) %>%
    ggplot(aes(x=time_ms, y=value, color=monitor)) + labs(x='time (ms)', y='weight (mV)', color='STDP method') +
    geom_line() + theme_bw() + scale_color_manual(values=c("#000000", color_map[4])) +
    guides(color=guide_legend(override.aes=list(linewidth=4))) +
    annotate("rect", xmin=b_rang[1], xmax=b_rang[2], ymin=10.75, ymax=11.36, alpha=0.3, fill=color_map[1]) +
    annotate("rect", xmin=b_rang[3], xmax=b_rang[4], ymin=10.75, ymax=11.36, alpha=0.3, fill=color_map[1]) +
    annotate("rect", xmin=b_rang[5], xmax=b_rang[6], ymin=10.75, ymax=11.36, alpha=0.3, fill=color_map[1]) +
    annotate("rect", xmin=b_rang[7], xmax=b_rang[8], ymin=10.75, ymax=11.36, alpha=0.3, fill=color_map[1]) +
    annotate("rect", xmin=b_rang[9], xmax=b_rang[10], ymin=10.75, ymax=11.36, alpha=0.3, fill=color_map[1]) +
    annotate("rect", xmin=b_rang[11], xmax=b_rang[12], ymin=10.75, ymax=11.36, alpha=0.3, fill=color_map[1]) +
    annotate(geom='text', x=b_rang[2] - (b_rang[2]-b_rang[1])/2, y=11.2, label='I') +
    annotate(geom='text', x=b_rang[4] - (b_rang[4]-b_rang[3])/2, y=11.2, label='II') +
    annotate(geom='text', x=b_rang[6] - (b_rang[6]-b_rang[5])/2, y=11.2, label='III') +
    annotate(geom='text', x=b_rang[8] - (b_rang[8]-b_rang[7])/2, y=11.2, label='IV') +
    annotate(geom='text', x=b_rang[10] - (b_rang[10]-b_rang[9])/2, y=11.2, label='V') +
    annotate(geom='text', x=b_rang[12] - (b_rang[12]-b_rang[11])/2, y=11.2, label='VI') +
    theme(text = element_text(size=16))

# Mean is taken over all connections, i.e. id
mse_w <- df_weights %>%
    filter(variable == 'w_plast') %>%
    mutate(value = value * 1000) %>%
    group_by(time_ms, id) %>%
    summarise(squared_error = diff(value)^2) %>%
    group_by(time_ms) %>%
    summarise(mse = mean(squared_error)) %>%
    ggplot(aes(x=time_ms, y=mse)) + labs(x='time (ms)', y='mean squared error') +
    geom_line() + theme_bw() +
    theme(panel.grid.minor=element_blank(), text = element_text(size=16)) +
    scale_x_continuous(breaks = scales::pretty_breaks(n = 3))
    
w_changes <- wrap_elements(w_trace + plot_annotation(title='A', theme=theme(text=element_text(size=16)))) / 
    (wrap_elements(mse_w + plot_annotation(title='B', theme=theme(text=element_text(size=16)))) +
     wrap_elements(ca_traces / df_post_ca_trace + plot_annotation(title='C', theme=theme(text=element_text(size=16)))))

ggsave(argv$dest, w_changes)
