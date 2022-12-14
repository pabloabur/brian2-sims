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
    stop("Folder with data folders must be provided")
}else{
    folder = args[1]
}

wd = getwd()
dir_list <- Sys.glob(file.path(wd, folder, 'win*', 'spikes.feather'))
sim_names <- map_chr(dir_list, ~str_extract(., "win[:digit:]*_bg[:digit:]*"))
spikes_list <- map(dir_list, read_feather)
dir_list <- Sys.glob(file.path(wd, folder, 'win*', 'metadata.json'))
metadata_list <- map(dir_list, fromJSON)

tsim <- map_dbl(metadata_list, ~as.double(str_sub(.$duration, 1, -2)))

# Measures
n_sample <- 1000
df_measures <- set_names(spikes_list, sim_names)
df_measures <- df_measures %>%
    map(~.x %>% group_by(layer) %>% slice_sample(n=n_sample))

rates <- map2(df_measures, tsim, ~mutate(.x, rate=map_int(t, length)/.y))
avg_rates <- rates %>%
    map(~.x %>% group_by(layer) %>% summarise(rate=mean(rate, na.rm=T)))
    

# R uses unbiased estimation, so result will be slightly different from numpy
cv <- df_measures %>%
    map(~.x %>% mutate(isi=map(t, diff))) %>%
    map(~.x %>% rowwise() %>% mutate(cv=sd(isi, na.rm=T)/mean(isi, na.rm=T)))
avg_cv <- cv %>%
    map(~.x %>% group_by(layer) %>% summarise(cv=mean(cv, na.rm=T)))

hist_bins <- map(tsim, ~seq(0, .*1000+3, by=3))
# For some reason there is an error if I name df_measures but not hist_bins
hist_bins <- set_names(hist_bins, sim_names)
syncs <- df_measures %>%
    imap(~.x %>%
         unnest_longer(t) %>%
         # Removing initial transients
         filter(t>=500) %>%
         group_by(layer) %>%
         summarise(sync_var=var(hist(t, hist_bins[[.y]], plot=F, right=F)$counts),
                   sync_mean=mean(hist(t, hist_bins[[.y]], plot=F, right=F)$counts, na.rm=T)) %>%
         rowwise() %>%
         mutate(sync=sync_var/sync_mean)
        )
syncs <- syncs %>% map(~.x %>% select(layer, sync))

df_measures <- list(avg_rates, avg_cv, syncs) %>%
    pmap(~reduce(list(..1, ..2, ..3), left_join, by="layer"))
df_measures <- df_measures %>%
    map(~.x %>%
        mutate(ai=if_else(rate<30 & sync<=8 & cv>=0.7 & cv<=1.2, T, F)))
    
ainess <- map_dbl(df_measures, ~100*sum(.$ai)/8)
ainess <- data.frame(ratio=ainess) %>%
    tibble::rownames_to_column(var='sim_type') %>%
    mutate(win=as.numeric(str_remove(str_extract(sim_type, 'win[:digit:]*'), 'win')),
           bg=as.numeric(str_remove(str_extract(sim_type, 'bg[:digit:]*'), 'bg')))

# get list of empty folders to set it to zero
dir_list <- Sys.glob(file.path(wd, folder, 'win*'))
empty_dirs <- map_lgl(dir_list, ~length(Sys.glob(file.path(., 'spikes.feather')))==0)
dir_list <- str_split(dir_list, '\\/') %>% map_chr(last)
empty_dirs <- dir_list[which(empty_dirs)]
weak_act <- data.frame(sim_type=empty_dirs) %>%
    mutate(ratio=0,
           win=as.numeric(str_remove(str_extract(sim_type, '(win)\\d*'), 'win')),
           bg=as.numeric(str_remove(str_extract(sim_type, '(bg)\\d*'), 'bg')))
ainess <- full_join(ainess, weak_act)
# contour plot was ignoring max value, so it is multiplied so that everyone is in range
fig <- ggplot(ainess, aes(x=win, y=bg, z=ratio/100)) +
    geom_contour_filled(breaks=c(seq(0, 1, .15), 1.05)) +
    scale_x_continuous(limits=c(66, 82), expand=c(0, 0)) +
    scale_y_continuous(limits=c(10, 70), expand=c(0, 0)) +
    scale_fill_manual(values=viridisLite::inferno(11),
                      guide=guide_colorsteps(title="%"))
ggsave(file.path(wd, folder, 'fig.png'), fig)
