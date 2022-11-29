library(ggplot2)
library(tidyr)
library(stringr)
library(dplyr)

# TODO sample before saving to disk?
a=arrow::read_feather('spikes.feather')
# TODO this needs to be loaded?
tsim <- 0.1 # TODO 60k-400/k for prot 1; something else for 2
# TODO both below also depend on protocol
min_time <- 0
max_time <- tsim*1000
# Approximately 20 per layer
# TODO sample proportion? I could do it when saving to disk and in here use proportion again
# NB:
# psample is used to get % of each layer for rates
# rates and cv taken from dataframe containing only nsamples from each layer
# sync from nsampled dataframe, so it's like sampling it likewise too
# AIness is just a result taken from each folder
# for protocol2, psample taken from each layer
n_sample <- 80

hist_bins <- seq(0, tsim*1000+3, by=3)

df_spk <- a %>%
    #slice_sample(n=n_sample) %>%
    unnest_longer(t) %>%
    mutate(laminae=str_sub(layer, 1, -2)) %>%
    # In case we want to change order
    #mutate(laminae=fct_relevel(laminae, c("L4", "L23", "L5", "L6"))) %>%
    filter(t>=min_time & t<=max_time)

rates <- a %>%
    rowwise() %>%
    mutate(rate=length(t)/tsim) %>%
    group_by(layer) %>%
    summarise(rate=mean(rate))

# R uses unbiased estimation, so result will probably be different from numpy
cv <- a %>%
    mutate(isi=map(t, diff))%>%
    rowwise()%>%
    mutate(cv=sd(isi)/mean(isi)) %>%
    group_by(layer) %>%
    summarise(cv=mean(cv))

# TODO first 500ms are ignored, that is why [166:]
sync_var <- df_spk %>% group_by(layer) %>% summarise(sync_var=var(hist(t, hist_bins, plot=F, right=F)$counts))
sync_mean <- df_spk %>% group_by(layer) %>% summarise(sync_mean=mean(hist(t, hist_bins, plot=F, right=F)$counts))
syncs <- full_join(sync_mean, sync_var, by="layer") %>%
    rowwise() %>%
    mutate(sync=sync_var/sync_mean)

# TODO protocol2; np.sum(np.split(count, tsim), axis=0) for each layer; the rest is similar
hist_bins2 <- seq(0, tsim*1000+.5, by=.5)

spks <- ggplot(df_spk, aes(x=t, y=i, color=layer)) +

         geom_point() + theme_bw() +
         theme(panel.grid.minor=element_blank(), panel.grid.major=element_blank()) +
         facet_grid(rows=vars(laminae), scales="free")

freq <- ggplot(rates, aes(rate, layer)) +
    geom_boxplot()
