library(ggplot2)
library(dplyr)
library(plotly)
library(jsonlite)
library(stringr)

library(argparser)
include('plots/parse_inputs.R')

wd = getwd()
data_path <- Sys.glob(file.path(wd, argv$source))

metadata <- fromJSON(file.path(data_path, "metadata.json"))

# Considered to be saved as second
tsim <- as.double(str_sub(metadata$duration, 1, -2))

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

ca_trace <- df_variables %>%
            filter(variable=='Ca') %>%
            ggplot(aes(t_ms, val, color=id)) +
            geom_line()
w_trace <- df_variables %>%
           filter(variable=='w_plast' & monitor=='stdp_in_w') %>%
           ggplot(aes(t_ms, val, color=id)) +
           geom_line()

raster <- ggplot(df_raster, aes(x=time_ms, y=id)) +
          geom_point(shape=20, size=1) + theme_bw()

#options("browser"="brave")
ggplotly(raster)
