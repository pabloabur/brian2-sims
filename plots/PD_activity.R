library(ggplot2)
library(tidyr)
library(stringr)
library(dplyr)
library(arrow)
library(purrr)

spikes <- read_feather('spikes.feather')

tsim <- 60 # TODO needs adjusting for full simulations

# Raster plot, protocol 1; somethig like .4s duration
min_time <- 59600
max_time <- tsim*1000
p_sample <- 0.025
df_raster <- spikes %>%
    group_by(layer) %>%
    slice_sample(prop=p_sample) %>%
    unnest_longer(t) %>%
    mutate(laminae=str_sub(layer, 1, -2)) %>%
    # In case we want to change order
    #mutate(laminae=fct_relevel(laminae, c("L4", "L23", "L5", "L6"))) %>%
    filter(t>=min_time & t<=max_time)

# Raster plot, protocol 2; interval can be selected
min_time <- 600
max_time <- 1000
df_raster2 <- spikes %>%
    group_by(layer) %>%
    slice_sample(prop=p_sample) %>%
    unnest_longer(t) %>%
    mutate(laminae=str_sub(layer, 1, -2)) %>%
    filter(t>=min_time & t<=max_time)

# TODO colored by exc/inh for each layer
# TODO remove ylabel, ticks, and adjust xlabel (size and name and in s)
spks <- ggplot(df_raster, aes(x=t, y=i, color=layer)) +
         geom_point() + theme_bw() +
         theme(panel.grid.minor=element_blank(), panel.grid.major=element_blank()) +
         facet_grid(rows=vars(laminae), scales="free")

# Measures
n_sample <- 1000
df_measures <- spikes %>%
    group_by(layer) %>%
    slice_sample(n=n_sample)

rates <- df_measures %>%
    rowwise() %>%
    mutate(rate=length(t)/tsim)

# R uses unbiased estimation, so result will be slightly different from numpy
cv <- df_measures %>%
    mutate(isi=map(t, diff))%>%
    rowwise()%>%
    mutate(cv=sd(isi, na.rm=T)/mean(isi, na.rm=T))

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

# AI landscape
# TODO AIness taken from multiple folders
# sys glob *.feather; files <- map(files, read_feather); map(gh_repos, ~map_dbl(.x, ~.x[["size"]])) %>% map(~max(.x))
avg_rates <- rates %>%
    group_by(layer) %>%
    summarise(rate=mean(rate, na.rm=T))
avg_cv <- cv %>%
    group_by(layer) %>%
    summarise(cv=mean(cv, na.rm=T))
df_measures <- reduce(list(avg_rates, avg_cv, select(syncs, layer, sync)), left_join, by="layer")
df_measures <- df_measures %>%
    mutate(ai=if_else(rate<30 & sync<=8 & cv>=0.7 & cv<=1.2, T, F))
ainess = 100*sum(df_measures$ai)/8.0

# TODO Styles for plots here
freq <- ggplot(rates, aes(rate, layer)) +
    geom_boxplot()
irregularity <- ggplot(df_measures, aes(cv, layer)) +
    geom_col()
synchronicity <- ggplot(df_measures, aes(sync, layer)) +
    geom_col()
#ggsave('test.png', freq)

# TODO protocol2; np.sum(np.split(count, tsim), axis=0) for each layer; the rest is similar
hist_bins2 <- seq(0, tsim*1000+.5, by=.5)
