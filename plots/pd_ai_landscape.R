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
library(argparser)
include('plots/parse_inputs.R')

wd = getwd()
dir_list <- Sys.glob(file.path(wd, argv$source, 'win*', 'spikes.feather'))
sim_names <- map_chr(dir_list, ~str_extract(., "win[:graph:]*_bg[:graph:]*"))
sim_names <- map_chr(sim_names, ~str_remove(., '/metadata.json'))
spikes_list <- map(dir_list, read_feather)
dir_list <- Sys.glob(file.path(wd, argv$source, 'win*', 'metadata.json'))
metadata_list <- map(dir_list, fromJSON)

tsim <- map_dbl(metadata_list, ~as.double(str_sub(.$duration, 1, -2)))

# Measures
n_sample <- 1000
df_measures <- set_names(spikes_list, sim_names)
df_measures <- df_measures %>%
    map(group_by, layer) %>%
    map(slice_sample, n=n_sample)

rates <- map2(df_measures, tsim, ~mutate(.x, rate=map_int(t, length)/.y))
avg_rates <- rates %>%
    map(group_by, layer) %>%
    map(summarise, rate=mean(rate, na.rm=T))


# R uses unbiased estimation, so result will be slightly different from numpy
cv <- df_measures %>%
    map(mutate, isi=map(t, diff)) %>%
    map(mutate, cv=map_dbl(isi, sd, na.rm=T)/map_dbl(isi, mean, na.rm=T)) %>%
    map(filter, !is.na(cv))
avg_cv <- cv %>%
    map(group_by, layer) %>%
    map(summarise, cv=mean(cv, na.rm=T))

hist_bins <- map(tsim, ~seq(0, .*1000+3, by=3))
# For some reason there is an error if I name df_measures but not hist_bins
hist_bins <- set_names(hist_bins, sim_names)
syncs <- df_measures %>%
    map(unnest_longer, t) %>%
     # Removing initial transients
    map(filter, t>=500) %>%
    imap(~.x %>%
         summarise(sync_var=var(hist(t, hist_bins[[.y]], plot=F, right=F)$counts, na.rm=T),
                   sync_mean=mean(hist(t, hist_bins[[.y]], plot=F, right=F)$counts, na.rm=T))
    ) %>%
    map(mutate, sync=sync_var/sync_mean)
syncs <- syncs %>%
    map(select, layer, sync)

df_measures <- list(avg_rates, avg_cv, syncs) %>%
    pmap(~reduce(list(..1, ..2, ..3), left_join, by="layer")) %>%
    map(filter, !is.na(cv))
df_measures <- df_measures %>%
    map(mutate, ai=if_else(rate<30 & sync<=8 & cv>=0.7 & cv<=1.2, T, F))

ainess <- map_dbl(df_measures, ~100*sum(.$ai)/8)
ainess <- data.frame(ratio=ainess) %>%
    tibble::rownames_to_column(var='sim_type') %>%
    mutate(win=as.numeric(str_extract(sim_type, '(?<=win)[:graph:]+(?=_)')),
           bg=as.numeric(str_extract(sim_type, '(?<=bg)[:digit:]+(?=\\/)')))

# get list of empty folders to set it to zero
dir_list <- Sys.glob(file.path(wd, argv$source, 'win*'))
empty_dirs <- map_lgl(dir_list, ~length(Sys.glob(file.path(., 'spikes.feather')))==0)
dir_list <- str_split(dir_list, '\\/') %>% map_chr(last)
empty_dirs <- dir_list[which(empty_dirs)]
weak_act <- data.frame(sim_type=empty_dirs) %>%
    mutate(ratio=0,
           win=as.numeric(str_remove(str_extract(sim_type, '(win)\\d*'), 'win')),
           bg=as.numeric(str_remove(str_extract(sim_type, '(bg)\\d*'), 'bg')))
ainess <- full_join(ainess, weak_act)
# contour plot was ignoring max value, so it is multiplied so that everyone is in range
fig <- ggplot(ainess, aes(x=win, y=bg)) +
    geom_raster(aes(fill=ratio), interpolate=T) +
    scale_x_continuous(expand=c(0, 0)) +
    scale_y_continuous(expand=c(0, 0)) +
    scale_fill_gradientn(colors=wes_palette('Zissou1'),
                         name='AI %') +
    labs(x='inhibitory weight (a.u)', y='background rate (Hz)')
ggsave(argv$dest, fig)
# To save plot as in thesis, I ran interactively for both int8 and fp8, and 
# saved it with fig <- figi | figf + plot_annotation(tag_levels='A');
# ggsave(argv$dest, ff, height=10, units='cm')
