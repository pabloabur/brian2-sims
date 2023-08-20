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
        ggplot(aes(x=rate, group=group, color=group, fill=group)) + theme_bw() +
        geom_histogram()

df_variables$id<-as.character(df_variables$id)
ca_trace <- df_variables %>%
            filter(variable=='Ca') %>%
            ggplot(aes(x=time_ms, y=value, group=id, color=id)) +
            geom_line()
g_trace <- df_variables %>%
           filter(variable=='g') %>%
           ggplot(aes(x=time_ms, y=value, group=id, color=id)) +
           geom_line()
w_trace <- df_variables %>%
           filter(variable=='w_plast' & monitor=='stdp_in_w') %>%
           ggplot(aes(x=time_ms, y=value, group=id, color=id)) +
           geom_line()

n_sample <- 1000
hist_bins <- seq(0, tsim*1000, by=3)
df_fano_factor <- df_raster %>%
               filter(id %in% sample(N_exc, size=n_sample))
var_binned_spikes = var(hist(df_fano_factor$time_ms, hist_bins, plot=F, right=F)$counts)
mean_binned_spikes = mean(hist(df_fano_factor$time_ms, hist_bins, plot=F, right=F)$counts)
fno = var_binned_spikes/mean_binned_spikes

#options("browser"="brave")
ggplotly(raster)
