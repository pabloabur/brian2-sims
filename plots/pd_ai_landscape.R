library(ggplot2)
library(patchwork)

library(arrow)
library(jsonlite)

library(tidyr)
library(stringr)
library(dplyr)
library(purrr)
library(forcats)
suppressPackageStartupMessages(library(magick))

library(wesanderson)
library(argparser)
include('plots/parse_inputs.R')

# create function to generate plot
ai_plot <- function(dir_list){
    sim_names <- map_chr(dir_list, ~str_extract(., "win[:graph:]*_bg[:graph:]*"))
    sim_names <- map_chr(sim_names, ~str_remove(., '/metadata.json'))
    spikes_list <- map(dir_list, read_feather)
    metadata_list <- map_chr(dir_list, \(x) str_replace(x, 'spikes.feather', 'metadata.json'))
    metadata_list <- map(metadata_list, fromJSON)

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
    rm(rates)
    gc()

    # R uses unbiased estimation, so result will be slightly different from numpy
    cv <- df_measures %>%
        map(mutate, isi=map(t, diff)) %>%
        map(mutate, cv=map_dbl(isi, sd, na.rm=T)/map_dbl(isi, mean, na.rm=T)) %>%
        map(filter, !is.na(cv))
    avg_cv <- cv %>%
        map(group_by, layer) %>%
        map(summarise, cv=mean(cv, na.rm=T))
    rm(cv)
    gc()

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
    rm(hist_bins)

    df_measures <- list(avg_rates, avg_cv, syncs) %>%
        pmap(~reduce(list(..1, ..2, ..3), left_join, by="layer")) %>%
        map(filter, !is.na(cv))
    rm(avg_rates, avg_cv, syncs)
    gc()
    df_measures <- df_measures %>%
        map(mutate, ai=if_else(rate<30 & sync<=8 & cv>=0.7 & cv<=1.2, T, F))

    ainess <- map_dbl(df_measures, ~100*sum(.$ai)/8)
    ainess <- data.frame(ratio=ainess) %>%
        tibble::rownames_to_column(var='sim_type') %>%
        mutate(win=as.numeric(str_extract(sim_type, '(?<=win)[:graph:]+(?=_)')),
               bg=as.numeric(str_extract(sim_type, '(?<=bg)[:digit:]+(?=\\/)')))
    rm(df_measures)
    gc()

    # get list of empty folders to set it to zero
    dir_list <- Sys.glob(file.path(str_remove(dir_list[1], 'win.*'), '*'))
    empty_dirs <- map_lgl(dir_list, ~length(Sys.glob(file.path(., '*')))==0)
    empty_dirs <- dir_list[which(empty_dirs)]
    empty_dirs <- empty_dirs[!str_detect(empty_dirs, 'description.txt')]
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
        scale_fill_continuous(low='black', high='white',
                             name='AI %') +
        theme(text = element_text(size=14)) +
        labs(x='inhibitory weight (a.u)', y='background rate (Hz)')

    rm(ainess)
    gc()
    return(fig)
}

# prefix was picked manually
prefix <- 'pd_async_irreg'
suffix <- '_int8'
wd = getwd()

print('Processing int8 data...')
dir_list <- Sys.glob(file.path(wd, argv$source, paste0(prefix, suffix), 'win*', 'spikes.feather'))
# Check is dir list is empty and throw an error if it is
if (identical(dir_list, character(0))){
    stop("Directory prefix must be the same, but suffix must be *_fp8 or *_int8")
}
figi <- ai_plot(dir_list)

print('Processing fp8 data...')
suffix <- '_fp8'
dir_list <- Sys.glob(file.path(wd, argv$source, paste0(prefix, suffix), 'win*', 'spikes.feather'))
# Check is dir list is empty and throw an error if it is
if (identical(dir_list, character(0))){
    stop("Directory prefix must be the same, but suffix must be *_fp8 or *_int8")
}
figf <- ai_plot(dir_list)

print('Processing original data...')
orig_PD_ai <- image_read("sim_data/ch2/pd_async_irreg_fp8/orig_pd_ai_landscape.png")
orig_PD_ai <- orig_PD_ai %>%
    image_ggplot()

fig <- (figi + figf) / orig_PD_ai +
    plot_layout(height=c(5, 10)) + plot_annotation(tag_levels='A')

ggsave(argv$dest, fig)
