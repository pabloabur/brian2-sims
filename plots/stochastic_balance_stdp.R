suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(arrow))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(plotly))
suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(stringr))
suppressPackageStartupMessages(library(purrr))
suppressPackageStartupMessages(library(patchwork))

library(wesanderson)
color_map <- wes_palette('FantasticFox1')

library(argparser)
include('plots/parse_inputs.R')
include('plots/minifloat_utils.R')

print_stats <- function(df_data, N_exc, tsim) {
    df_base_stats <- df_data %>%
                     filter(id < N_exc+1) %>%
                     group_by(id) %>%
                     summarise(rate=n()/tsim,
                               mean_isi=mean(diff(time_ms)),
                               sd_isi=sd(diff(time_ms)))

    mean_rate = mean(df_base_stats$rate)
    print("Mean rate: ")
    print(mean_rate)

    mean_cv <- mean(df_base_stats$sd_isi/df_base_stats$mean_isi, na.rm=T)
    print("Mean CV: ")
    print(mean_cv)

    n_sample <- 1000
    print("Calculating total Fano factor")
    df_fano_factor <- df_data %>%
                           filter(id %in% sample(N_exc, size=n_sample))
    hist_bins <- seq(min(df_fano_factor$time_ms),
                     max(df_fano_factor$time_ms),
                     by=3)
    hist_bins <- c(hist_bins, nth(hist_bins, -1) + 3)
    hist_counts = hist(df_fano_factor$time_ms, hist_bins, plot=F, right=F)$counts
    fno = var(hist_counts)/mean(hist_counts)
    print("Fano factor: ")
    print(fno)

    print("Calculating Fano factor of last 50 seconds")
    df_fano_factor_init <- df_data %>%
                           filter(time_ms>0 & time_ms<50000) %>%
                           filter(id %in% sample(N_exc, size=n_sample))
    hist_bins <- seq(min(df_fano_factor_init$time_ms),
                     max(df_fano_factor_init$time_ms),
                     by=3)
    hist_bins <- c(hist_bins, nth(hist_bins, -1) + 3)
    hist_counts = hist(df_fano_factor_init$time_ms, hist_bins, plot=F, right=F)$counts
    fno = var(hist_counts)/mean(hist_counts)
    print("Fano factor in the first 50 seconds: ")
    print(fno)

    df_fano_factor_final <- df_data %>%
                            filter(time_ms>(tsim - 50)*1000 & time_ms<tsim*1000) %>%
                            filter(id %in% sample(N_exc, size=n_sample))
    hist_bins <- seq(min(df_fano_factor_final$time_ms),
                     max(df_fano_factor_final$time_ms),
                     by=3)
    hist_bins <- c(hist_bins, nth(hist_bins, -1) + 3)
    hist_counts = hist(df_fano_factor_final$time_ms, hist_bins, plot=F, right=F)$counts
    fno = var(hist_counts)/mean(hist_counts)
    print("Fano factor in the last 50 seconds: ")
    print(fno)
}

plot_weights <- function(df_data, bin_width, subset=NULL){
    color_map <- wes_palette('GrandBudapest1')

    ds_weights <- open_dataset(file.path(df_data, 'obj_vars.feather'),
                               format="arrow")
    if (is.null(subset)) {
        weights <- (ds_weights %>% select(w_plast) %>% collect())$w_plast
    } else {
        weights <- (ds_weights %>% filter(j==subset) %>% select(w_plast) %>%
                    collect())$w_plast
    }

    max_w <- max(weights)
    min_w <- min(weights)
    w_hist <- hist(weights, breaks=seq(min_w, max_w), plot=F)
    rm(weights)
    gc()
    w_hist$breaks <- map_dbl(w_hist$breaks, minifloat2decimal)
    w_hist$breaks <- head(w_hist$breaks, -1)
    df_weights <- data.frame(count = w_hist$counts,
                             mids = w_hist$breaks)
    # using density does not work because area is not 1
    df_weights$count <- df_weights$count/sum(df_weights$count)

    w_distribution <- df_weights %>%
        ggplot(aes(x=mids, y=count)) +
        geom_col(width=bin_width, fill=color_map[2], color=color_map[2]) + 
        theme_bw() + labs(x='weights (a.u.)', y='fraction of synapses')

    return(w_distribution)
}

wd = getwd()
data_path <- Sys.glob(file.path(wd, argv$source))
metadata <- map(file.path(data_path, "metadata.json"), fromJSON)
tsim <- map_int(metadata, \(x) as.double(str_sub(x$duration, 1, -2)))
N_exc <- map_int(metadata, \(x) x$N_exc)

state_vars <- read.csv(file.path(data_path, "output_vars.csv"))
state_vars$value <- map_dbl(state_vars$value, minifloat2decimal)
min_time <- 0
variables <- state_vars %>%
    filter(time_ms>=min_time & time_ms<=1e3+min_time) %>%
    ggplot(aes(x=time_ms, y=value, color=variable)) +
    geom_line() + theme_bw()

w_distribution <- plot_weights(data_path, bin_width=0.64)
incoming_w_j0 <- plot_weights(data_path, bin_width=0.64, 0)
incoming_w_j10 <- plot_weights(data_path, bin_width=0.64, 10000)

event_data <- read.csv(file.path(data_path, 'events_spikes.csv'))
fetches <- event_data %>%
    ggplot(aes(x=time_ms, y=num_events)) + geom_line() +
    theme_bw() + labs(x='time (ms)', y='# Active neurons')

df_raster <- read.csv(file.path(data_path, 'output_spikes.csv'))
print("Statistics with bimodal experiment: ")
print_stats(df_raster, N_exc, tsim)

df_rates_init <- df_raster %>%
                 filter(time_ms>0 & time_ms<2000) %>%
                 group_by(id) %>%
                 summarise(rate=n()/2)
df_rates_final <- df_raster %>%
                  filter(time_ms>(tsim - 2)*1000 & time_ms<tsim*1000) %>%
                  group_by(id) %>%
                  summarise(rate=n()/2)

df_rates_final$group <- 'final'
df_rates_init$group <- 'initial'
freq <- bind_rows(df_rates_init, df_rates_final) %>%
        ggplot(aes(x=rate, color=group, fill=group)) + theme_bw() +
        geom_histogram(alpha=.5) + scale_color_manual(values=color_map) + 
        scale_fill_manual(values=color_map) + theme(legend.position = c(0.8, 0.8)) +
        labs(x='firing rates (Hz)', y='count', fill=element_blank(), color=element_blank()) +
        scale_y_log10()

fig_events <- fetches + freq + plot_annotation(tag_levels='A')
fig_vars <- variables + w_distribution + plot_annotation(tag_levels='A')

ggsave(str_replace(argv$dest, '.png', '_events.png'), fig_events)
ggsave(str_replace(argv$dest, '.png', '_variables.png'), fig_vars)
