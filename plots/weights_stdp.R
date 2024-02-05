suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(patchwork))
suppressPackageStartupMessages(library(latex2exp))
suppressPackageStartupMessages(library(stringr))

library(wesanderson)
color_map <- wes_palette('GrandBudapest1')

library(argparser)
include('plots/parse_inputs.R')

sim_dir <- 'kernel_sim'
df_weights <- read.csv(file.path(argv$source, sim_dir, 'synapse_vars.csv'))
kernel <- df_weights %>%
    filter(time_ms == max(df_weights$time_ms) &
                          variable=='w_plast') %>%
    mutate(value = value * 1000) %>%
    mutate(id = id - 29.5) %>%
    mutate(monitor = str_replace_all(monitor, c('ref_statemon_post_synapse' = 'Original',
                                                'statemon_post_synapse' = 'Proposed'))) %>%
    ggplot(aes(x=id, y=value, color=monitor)) + geom_line() + theme_bw() +
    labs(x=TeX(r'($\Delta$\;t (ms))'), y='weight (mV)', color=element_blank()) +
    guides(color=guide_legend(override.aes=list(linewidth=4))) +
    scale_color_manual(values=color_map) +
    theme(legend.position = c(0.75, 0.2),
          text = element_text(size=16),
          legend.text=element_text(size=10))

kernel_mse <- df_weights %>%
    filter(variable == 'w_plast') %>%
    mutate(value = value * 1000) %>%
    group_by(time_ms, id) %>%
    summarise(squared_error = diff(value)^2) %>%
    group_by(time_ms) %>%
    summarise(mse = mean(squared_error)) %>%
    ggplot(aes(x=time_ms, y=mse)) + labs(x='time (ms)', y='mean squared error') +
    geom_line() + theme_bw() +
    theme(panel.grid.minor=element_blank(), text = element_text(size=16))
    
sim_dir <- 'distribution_sim'
df_weights <- read.csv(file.path(argv$source, sim_dir, 'synapse_vars_weights.csv'))
df_weights$w_plast <- df_weights$w_plast * 1000
w_distribution <- df_weights %>%
    filter(label=='Proposed') %>%
    ggplot(aes(x=w_plast, color=label, fill=label)) + geom_histogram(alpha=0.8) +
    theme_bw() + theme(legend.position="none", text = element_text(size=16)) +
    labs(x='weights (mV)', color=element_blank(), fill=element_blank()) +
    scale_color_manual(values=color_map[2]) +
    scale_fill_manual(values=color_map[2])

w_profiles <- ((kernel + kernel_mse) / w_distribution) +
    plot_annotation(tag_levels='A')
ggsave(argv$dest, w_profiles)
