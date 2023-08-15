library(ggplot2)
library(dplyr)
library(plotly)
library(jsonlite)
library(stringr)
library(purrr)
library(patchwork)

library(argparser)
include('plots/parse_inputs.R')

wd = getwd()
data_path <- Sys.glob(file.path(wd, argv$source))

metadata <- fromJSON(file.path(data_path, "metadata.json"))

# Considered to be saved as second
tsim <- as.double(str_sub(metadata$duration, 1, -2))
N_exc <- metadata$N_exc

df_raster <- read.csv(file.path(data_path, 'output_spikes.csv'))
df_variables <- read.csv(file.path(data_path, 'output_vars.csv'))

df_rates_total <- df_raster %>%
                  group_by(id) %>%
                  summarise(rate=n()/tsim)
df_rates_init <- df_raster %>%
                 filter(time_ms>0 & time_ms<2000) %>%
                 group_by(id) %>%
                 summarise(rate=n()/2)
df_rates_final <- df_raster %>%
                  filter(time_ms>(tsim - 2)*1000 & time_ms<tsim*1000) %>%
                  group_by(id) %>%
                  summarise(rate=n()/2)

freq_init <- df_rates_init %>%
        ggplot(aes(rate)) + theme_bw() +
        geom_histogram()
freq_final <- df_rates_final %>%
        ggplot(aes(rate)) + theme_bw() +
        geom_histogram()
freq_total <- df_rates_total %>%
        ggplot(aes(rate)) + theme_bw() +
        geom_histogram()

# TODO color=id is wrong, look at filter id==1 or ggplotly
ca_trace <- df_variables %>%
            filter(variable=='Ca') %>%
            ggplot(aes(time_ms, value, color=id)) +
            geom_line()
g_trace <- df_variables %>%
           filter(variable=='g') %>%
           ggplot(aes(time_ms, value, color=id)) +
           geom_line()
w_trace <- df_variables %>%
           filter(variable=='w_plast' & monitor=='stdp_in_w') %>%
           ggplot(aes(time_ms, value, color=id)) +
           geom_line()

# TODO #1: output from histogram over time, #2: same, ..., #N...;
       # then they are summed together, resulting in same-length output
       # then valvulate average, and mean for fano
n_sample <- 1000
hist_bins <- seq(0, tsim*1000, by=3)
df_fano_factor <- df_raster %>%
               filter(id %in% sample(N_exc, size=n_sample))
var_binned_spikes = var(hist(df_fano_factor$time_ms, hist_bins, plot=F, right=F)$counts)
mean_binned_spikes = mean(hist(df_fano_factor$time_ms, hist_bins, plot=F, right=F)$counts)
fno = var_binned_spikes/mean_binned_spikes

#options("browser"="brave")
ggplotly(raster)
