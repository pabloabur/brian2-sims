suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(purrr))
suppressPackageStartupMessages(library(stringr))
suppressPackageStartupMessages(library(wesanderson))
suppressPackageStartupMessages(library(patchwork))
suppressPackageStartupMessages(library(latex2exp))

library(argparser)
include('plots/parse_inputs.R')
include('plots/minifloat_utils.R')

ca_trace <- function(df_data, color_map){
    fig <- df_data %>%
        mutate(value=map_dbl(value, minifloat2decimal)) %>%
        group_by(time_ms) %>%
        mutate(avg_value=mean(value)) %>%
        ggplot(aes(x=time_ms, y=avg_value)) + geom_line(color=color_map[4]) +
        theme_bw() + labs(x='time (ms)', y=TeX(r'(average $x$ (a.u.))'))

    return(fig)
}

stats <- function(df_data, color_map, tsim){
    df_stats_init <- df_data %>%
        group_by(id) %>%
        filter(time_ms<1000) %>%
        summarise(rate=n()/1,
               mean_isi=mean(diff(time_ms), na.rm=T),
               sd_isi=sd(diff(time_ms), na.rm=T),
               cv=sd_isi/mean_isi)
    df_stats_final <- df_data %>%
        group_by(id) %>%
        filter(time_ms>(1000*tsim-1000)) %>%
        summarise(rate=n()/1,
               mean_isi=mean(diff(time_ms), na.rm=T),
               sd_isi=sd(diff(time_ms), na.rm=T),
               cv=sd_isi/mean_isi)

    rate_init <- df_stats_init %>%
        ggplot(aes(x=rate)) + geom_histogram(fill=color_map[4], binwidth=1) +
        theme_bw() + labs(x='firing rate (Hz)', y='count')
    cv_init <- df_stats_init %>%
        ggplot(aes(x=cv)) + geom_histogram(fill=color_map[4], binwidth=0.1) +
        theme_bw() + labs(x='ISI CV', y='count')
    rate_final <- df_stats_final %>%
        ggplot(aes(x=rate)) + geom_histogram(fill=color_map[4], binwidth=1) +
        theme_bw() + labs(x='firing rate (Hz)', y='count')
    cv_final <- df_stats_final %>%
        ggplot(aes(x=cv)) + geom_histogram(fill=color_map[4], binwidth=0.1) +
        theme_bw() + labs(x='ISI CV', y='count')

    return(list(rate_init, cv_init, rate_final, cv_final, df_stats_init, df_stats_final))
}

color_map <- wes_palette('Rushmore1')

wd = getwd()
data_path <- Sys.glob(file.path(wd, argv$source, "*"))
metadata <- map(file.path(data_path, "metadata.json"), fromJSON)
protocol <- map_int(metadata, \(x) x$protocol)
tsim <- map_dbl(metadata, \(x) as.double(str_sub(x$duration, 1, -2)))

####### Processing trial with high learning rate
sel_dir <- match(1, protocol)
state_vars <- read.csv(file.path(data_path[sel_dir], "state_vars.csv"))
exc_spikes <- read.csv(file.path(data_path[sel_dir], "spikes_exc.csv"))

ca_trace_high <- ca_trace(state_vars, color_map)

# get small one for comparison
state_vars <- read.csv(file.path(data_path[match(2, protocol)], "state_vars.csv"))
ca_trace_small <- ca_trace(state_vars, color_map)

exc_spikes <- read.csv(file.path(data_path[sel_dir], "spikes_exc.csv"))
stats_figs <- stats(exc_spikes, color_map, tsim[sel_dir])

####### Processing trial with small learning rate
sel_dir <- match(2, protocol)
exc_spikes <- read.csv(file.path(data_path[sel_dir], "spikes_exc.csv"))
inh_weights <- read.csv(file.path(data_path[sel_dir], "weights.csv"))

w_traces <- inh_weights %>%
    mutate(value=map_dbl(value, minifloat2decimal)) %>%
    mutate(time_ms=time_ms/1000) %>%
    ggplot(aes(x=value, fill=time_ms, group=time_ms)) +
    geom_histogram(alpha=0.7) +
    scale_fill_gradientn(colors=color_map) +
    theme_bw() + labs(x='weight (a.u.)', y='count', fill='time (s)') +
    theme(legend.position = c(0.80, 0.8))

raster <- function(df_data, color_map, time_slice){
    fig <- df_data %>%
        filter(time_ms>time_slice[1] & time_ms<time_slice[2]) %>%
        ggplot(aes(x=time_ms, y=id)) +
        geom_point(shape=20, size=0.05, alpha=0.2, color=color_map[4]) +
        theme_bw() +
        theme(panel.grid.minor=element_blank(),
              panel.grid.major=element_blank()) +
        scale_color_manual(values=color_map[4]) +
        labs(x='time (ms)', y='neuron id')

    return(fig)
}

raster_init <- raster(exc_spikes, color_map, c(0, 1000))
raster_final <- raster(exc_spikes, color_map, c(1000*tsim[sel_dir]-1000, 1000*tsim[sel_dir]))

stats_figs <- stats(exc_spikes, color_map, tsim[sel_dir])

RS <- stats_figs[[5]]$cv
AI <- stats_figs[[6]]$cv
print(t.test(RS, AI))

fig_high_protocol <- wrap_elements(ca_trace_high + ca_trace_small +
                                   plot_annotation(title='A')) /
                     (wrap_elements(stats_figs[[1]] + plot_annotation(title='B')) |
                     wrap_elements(stats_figs[[2]] + plot_annotation(title='C')))

fig_small_protocol <- (wrap_elements(raster_init + stats_figs[[1]] + stats_figs[[2]] +
                                     plot_annotation(title='A')) /
                       wrap_elements(raster_final + stats_figs[[3]] + stats_figs[[4]] +
                                     plot_annotation(title='B'))) |
                       wrap_elements(w_traces + plot_annotation(title='C'))

ggsave(str_replace(argv$dest, '.png', '_high_eta.png'), fig_high_protocol)
ggsave(str_replace(argv$dest, '.png', '_small_eta.png'), fig_small_protocol)
