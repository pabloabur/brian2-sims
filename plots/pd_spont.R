library(ggplot2)
library(patchwork)

library(arrow)
library(jsonlite)

library(tidyr)
library(stringr)
library(dplyr)
library(purrr)
library(forcats)

args = commandArgs(trailingOnly=T)
if (length(args)==0){
    stop("Folder with data must be provided")
}else{
    folder = args[1]
}

wd = getwd()
dir <- Sys.glob(file.path(wd, folder))

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

raster <- ggplot(df_raster, aes(x=t, y=i, color=type)) +
         geom_point(shape=20, size=1) + theme_bw() +
         theme(panel.grid.minor=element_blank(),
               panel.grid.major=element_blank(),
               axis.text.y=element_blank(),
               axis.ticks.y=element_blank()) +
         guides(color=guide_legend(override.aes=list(size=5))) +
         facet_grid(rows=vars(laminae), scales="free") +
         labs(x='time (ms)', y=element_blank())

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
    ggplot(aes(rate, layer)) + theme_bw() +
    theme(panel.grid.minor=element_blank(),
          panel.grid.major=element_blank()) +
    labs(x='firing rates (Hz)', y=element_blank()) +
    geom_boxplot() +
    stat_summary(fun='mean', geom='point', shape=2)

irregularity <- avg_cv %>%
    mutate(layer=fct_rev(layer)) %>%
    ggplot(aes(cv, layer)) +
    theme_bw() + geom_col() +
    theme(panel.grid.minor=element_blank(),
          panel.grid.major=element_blank()) +
    labs(x='irregularity', y=element_blank())

synchrony <- syncs %>%
    mutate(layer=fct_rev(layer)) %>%
    ggplot(aes(sync, layer)) +
    theme_bw() + geom_col() +
    theme(panel.grid.minor=element_blank(),
          panel.grid.major=element_blank()) +
    labs(x='synchrony', y=element_blank())

fig <- (raster | (freq / irregularity / synchrony)) + plot_annotation(tag_levels='A')
ggsave(file.path(dir, 'fig.png'), fig)
