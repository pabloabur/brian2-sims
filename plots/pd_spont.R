library(ggplot2)
library(patchwork)

library(arrow)
library(jsonlite)

library(tidyr)
library(stringr)
library(dplyr)
library(purrr)
library(forcats)
library(wesanderson)
suppressPackageStartupMessages(library(magick))

library(argparser)
include('plots/parse_inputs.R')

color_map <- wes_palette('Moonrise1')

wd = getwd()
dir <- Sys.glob(file.path(wd, argv$source))

spikes <- read_feather(file.path(dir, "spikes.feather"))
metadata <- fromJSON(file.path(dir, "metadata.json"))

tsim <- as.double(str_sub(metadata$duration, 1, -2))

# Units considered in ms
min_time <- tsim*1000 - 400
max_time <- tsim*1000
p_sample <- 0.025
df_raster <- spikes %>%
    group_by(layer) %>%
    slice_sample(prop=p_sample) %>%
    unnest_longer(t) %>%
    mutate(laminae=str_sub(layer, 1, -2)) %>%
    mutate(type=if_else(str_sub(layer, -1, -1)=='e', 'Exc', 'Inh')) %>%
    filter(t>=min_time & t<=max_time)

raster <- df_raster %>%
    mutate(t=t/1000) %>%
    ggplot(aes(x=t, y=i, color=type)) +
    geom_point(shape=20, size=1) + theme_bw() +
    theme(panel.grid.minor=element_blank(),
          panel.grid.major=element_blank(),
          axis.text.y=element_blank(),
          axis.ticks.y=element_blank(),
          legend.position="top",
          text = element_text(size=16)) +
    guides(color=guide_legend(override.aes=list(size=5))) +
    facet_grid(rows=vars(laminae), scales="free") +
    labs(x='time (s)', y=element_blank()) +
    scale_color_manual(values=color_map[c(4, 2)]) +
    scale_x_continuous(breaks = scales::pretty_breaks(n = 3))

# Measures
n_sample <- 1000
df_measures <- spikes %>%
    group_by(layer) %>%
    slice_sample(n=n_sample)

rates <- df_measures %>%
    rowwise() %>%
    mutate(rate=length(t)/tsim)
avg_rates <- rates %>%
    group_by(layer) %>%
    summarise(rate=mean(rate, na.rm=T))

# R uses unbiased estimation, so result will be slightly different from numpy
cv <- df_measures %>%
    mutate(isi=map(t, diff))%>%
    rowwise()%>%
    mutate(cv=sd(isi, na.rm=T)/mean(isi, na.rm=T))
avg_cv <- cv %>%
    group_by(layer) %>%
    summarise(cv=mean(cv, na.rm=T))

hist_bins <- seq(0, tsim*1000+3, by=3)
sync_var <- df_measures %>%
    unnest_longer(t) %>%
    # Removing initial transients
    filter(t>=500) %>%
    group_by(layer) %>%
    summarise(sync_var=var(hist(t, hist_bins, plot=F, right=F)$counts))
sync_mean <- df_measures %>%
    unnest_longer(t) %>%
    # Removing initial transients
    filter(t>=500) %>%
    group_by(layer) %>%
    summarise(sync_mean=mean(hist(t, hist_bins, plot=F, right=F)$counts, na.rm=T))
syncs <- full_join(sync_mean, sync_var, by="layer") %>%
    rowwise() %>%
    mutate(sync=sync_var/sync_mean)

freq <- rates %>%
    mutate(layer=fct_rev(layer)) %>%
    mutate(type=if_else(str_sub(layer, -1, -1)=='e', 'Exc', 'Inh')) %>%
    ggplot(aes(rate, layer, fill=type)) + theme_bw() +
    theme(panel.grid.minor=element_blank(),
          panel.grid.major=element_blank(),
          legend.position="none",
          panel.border=element_blank(),
          axis.line.x = element_line(color='black'),
          axis.line.y = element_line(color='black'),
          text = element_text(size=16)) +
    labs(x='firing rates (Hz)', y=element_blank()) +
    geom_boxplot(outlier.colour="grey",
                 outlier.fill="grey",
                 outlier.shape=3,
                 outlier.size=1) +
    stat_summary(fun='mean', geom='point', shape="\u2B50", color='white', size=5) +
    scale_fill_manual(values=c('Exc'='grey36', 'Inh'='grey')) +
    guides(y=guide_axis(angle=45))

irregularity <- avg_cv %>%
    mutate(layer=fct_rev(layer)) %>%
    mutate(type=if_else(str_sub(layer, -1, -1)=='e', 'Exc', 'Inh')) %>%
    ggplot(aes(cv, layer, fill=type)) +
    theme_bw() + geom_col(color='black', width=0.5) +
    theme(panel.grid.minor=element_blank(),
          panel.grid.major=element_blank(),
          legend.position="none",
          panel.border=element_blank(),
          axis.line.x = element_line(color='black'),
          axis.line.y = element_line(color='black'),
          text = element_text(size=16)) +
    labs(x='irregularity', y=element_blank()) +
    scale_fill_manual(values=c('Exc'='grey36', 'Inh'='grey')) +
    guides(y=guide_axis(angle=45)) +
    scale_x_continuous(breaks = scales::pretty_breaks(n = 3))

synchrony <- syncs %>%
    mutate(layer=fct_rev(layer)) %>%
    mutate(type=if_else(str_sub(layer, -1, -1)=='e', 'Exc', 'Inh')) %>%
    ggplot(aes(sync, layer, fill=type)) +
    theme_bw() + geom_col(color='black', width=0.5) +
    theme(panel.grid.minor=element_blank(),
          panel.grid.major=element_blank(),
          legend.position="none",
          panel.border=element_blank(),
          axis.line.x = element_line(color='black'),
          axis.line.y = element_line(color='black'),
          text = element_text(size=16)) +
    labs(x='synchrony', y=element_blank()) +
    scale_fill_manual(values=c('Exc'='grey36', 'Inh'='grey')) +
    guides(y=guide_axis(angle=45)) +
    scale_x_continuous(breaks = scales::pretty_breaks(n = 4))

orig_PD_stats <- image_read("sim_data/ch2/pd_async_irreg_fp8/orig_pd_stats.png")
orig_PD_stats <- orig_PD_stats %>%
    image_ggplot()

fig <- wrap_elements(raster + plot_annotation(title='A', theme=theme(text=element_text(size=16)))) |
       wrap_elements(freq / irregularity / synchrony + plot_annotation(title='B', theme=theme(text=element_text(size=16)))) |
       wrap_elements(orig_PD_stats + plot_annotation(title='C', theme=theme(text=element_text(size=16))))
ggsave(argv$dest, fig)
