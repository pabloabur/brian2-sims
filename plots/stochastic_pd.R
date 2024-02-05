suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(tidyr))
suppressPackageStartupMessages(library(stringr))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(arrow))
suppressPackageStartupMessages(library(purrr))
suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(wesanderson))
suppressPackageStartupMessages(library(plotly))
suppressPackageStartupMessages(library(patchwork))
suppressPackageStartupMessages(library(forcats))
suppressPackageStartupMessages(library(xtable))

library(argparser)
include('plots/parse_inputs.R')
include('plots/minifloat_utils.R')

color_map <- wes_palette('FantasticFox1')

wd = getwd()
data_path <- Sys.glob(file.path(wd, argv$source, "*"))
data_path <- data_path[str_detect(data_path, ".png", negate=T)]

metadata <- map(file.path(data_path, "metadata.json"), fromJSON)
durations = map_chr(metadata, \(x) x$duration)

# Simulation of spontaneous activity
sim_id <- map_int(metadata, \(x) x$protocol) == 1
spikes <- read_feather(file.path(data_path[sim_id], "spikes.feather"))
tsim <- as.double(str_sub(durations[sim_id], 1, -2))

n_sample <- 1000
df_measures <- spikes %>%
    group_by(layer) %>%
    slice_sample(n=n_sample)

rates <- df_measures %>%
    rowwise() %>%
    mutate(rate=length(t)/tsim)

freq <- rates %>%
    mutate(layer=fct_rev(layer)) %>%
    ggplot(aes(rate, layer)) + theme_bw() +
    theme(panel.grid.minor=element_blank(),
          panel.grid.major=element_blank(),
          text=element_text(size=16)) +
    labs(x='firing rates (Hz)', y=element_blank()) +
    geom_boxplot() +
    stat_summary(fun='mean', geom='point', shape=2)

avg_rates <- rates %>%
    group_by(layer) %>%
    summarise(rate=mean(rate, na.rm=T))
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
df_measures <- bind_cols(avg_rates, avg_cv, syncs) %>%
    mutate(ai=if_else(rate<30 & sync<=8 & cv>=0.7 & cv<=1.2, T, F)) %>%
    select(layer...1, rate, cv, sync, ai)
names(df_measures)[1] <- 'layer'

# Simulation of thalamic input
sim_id <- map_int(metadata, \(x) x$protocol) == 2
spikes <- read_feather(file.path(data_path[sim_id], "spikes.feather"))
state_vars <- read_feather(file.path(data_path[sim_id], "synapse_vars.feather"))
tsim <- as.double(str_sub(durations[sim_id], 1, -2))

state_vars <- state_vars %>%
    mutate(layer=case_when(
        id==0 ~ 'L23',
        id==20683 ~ 'L23',
        id==26517 ~ 'L4',
        id==48432 ~ 'L4',
        id==53911 ~ 'L5',
        id==58761 ~ 'L5',
        id==59826 ~ 'L6',
        id==74221 ~ 'L6'
        )) %>%
    mutate(type=case_when(
        id==0 ~ 'Exc',
        id==20683 ~ 'Inh',
        id==26517 ~ 'Exc',
        id==48432 ~ 'Inh',
        id==53911 ~ 'Exc',
        id==58761 ~ 'Inh',
        id==59826 ~ 'Exc',
        id==74221 ~ 'Inh'))
state_vars$value <- map_dbl(state_vars$value, minifloat2decimal)

# Units considered in ms. Input comes around 700ms after each trial
trials <- seq(tsim)
min_time <- trials[1]*1000 - 330
max_time <- trials[1]*1000 - 270
p_sample <- 0.025
df_raster <- spikes %>%
    group_by(layer) %>%
    slice_sample(prop=p_sample) %>%
    unnest_longer(t) %>%
    mutate(laminae=str_sub(layer, 1, -2)) %>%
    mutate(type=if_else(str_sub(layer, -1, -1)=='e', 'Exc', 'Inh'))

histograms <- df_raster %>%
    mutate(t=t%%1000) %>%
    ggplot(aes(x=t, color=type)) + geom_step(stat='bin', bins=50) +
    guides(color=guide_legend(override.aes=list(size=4))) +
    facet_grid(rows=vars(laminae), scales='free') + theme_bw() +
    theme(legend.position = 'bottom', text=element_text(size=16)) +
    scale_color_manual(values=color_map[c(1, 3)]) +
    guides(color=guide_legend(override.aes=list(linewidth=5))) +
    labs(x='time (ms)') + xlim(670, 730)

variables <- state_vars %>%
    filter(time_ms>=min_time & time_ms<=1e3+min_time) %>%
    mutate(variable = str_replace(variable, "Ca", "x")) %>%
    mutate(variable = str_replace(variable, "g", "PSP")) %>%
    mutate(time_ms=time_ms/1000) %>%
    ggplot(aes(x=time_ms, y=value, color=variable)) +
    geom_line() + theme_bw() +
    facet_grid(layer ~ type, scales="free") +
    theme(legend.position = 'bottom',
          strip.text.y = element_blank(),
          text=element_text(size=16)) +
    labs(x='time (s)', y='magnitude (a.u.)') + scale_color_manual(values=color_map) +
    guides(color=guide_legend(override.aes=list(linewidth=5)))

fig_thal <- histograms + variables + plot_annotation(tag_levels='A') +
    plot_layout(widths = c(1.3, 1.7))

print(xtable(df_measures))
ggsave(str_replace(argv$dest, '.png', '_thal.png'), fig_thal)
ggsave(str_replace(argv$dest, '.png', '_freq.png'), freq)
