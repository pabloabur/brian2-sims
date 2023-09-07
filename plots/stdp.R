suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(plotly))
suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(stringr))
suppressPackageStartupMessages(library(purrr))
suppressPackageStartupMessages(library(patchwork))

library(wesanderson)
color_map <- wes_palette('GrandBudapest1')

library(argparser)
include('plots/parse_inputs.R')

wd = getwd()
# TODO
#argv$source <- 'bal_stdp_cudadgx3'
data_path <- Sys.glob(file.path(wd, argv$source))

metadata <- fromJSON(file.path(data_path, "metadata.json"))

# Considered to be saved as second
tsim <- as.double(str_sub(metadata$duration, 1, -2))
N_exc <- metadata$N_exc

df_raster <- read.csv(file.path(data_path, 'output_spikes.csv'))
df_variables <- read.csv(file.path(data_path, 'output_vars.csv'))

# TODO
#g_trace <- df_variables %>%
#           filter(variable=='g') %>%
#           ggplot(aes(x=time_ms, y=value, group=id, color=id)) +
#           geom_line()
#vm_trace <- df_variables %>%
#           filter(variable=='Vm') %>%
#           ggplot(aes(x=time_ms, y=value, group=id, color=id)) +
#           geom_line()
#options("browser"="brave")
#ggplotly(vm_trace)
#ggplotly(g_trace)

df_base_stats <- df_raster %>%
                 filter(id < N_exc+1) %>%
                 group_by(id) %>%
                 summarise(rate=n()/tsim,
                           mean_isi=mean(diff(time_ms)),
                           sd_isi=sd(diff(time_ms)))

mean_rate = mean(df_base_stats$rate)
print("Mean rate: ")
print(mean_rate)

print("Plotting rates")
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
        geom_histogram(alpha=.3)
ggsave(str_replace(argv$dest, '.png', '_freq.png'), freq)

mean_cv <- mean(df_base_stats$sd_isi/df_base_stats$mean_isi, na.rm=T)
print("Mean CV: ")
print(mean_cv)

print("Plotting traces")
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
traces <- ca_trace / g_trace / w_trace + plot_annotation(tag_levels = 'A')
ggsave(str_replace(argv$dest, '.png', '_traces.png'), traces)

n_sample <- 1000
print("Calculating total Fano factor")
df_fano_factor <- df_raster %>%
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
df_fano_factor_init <- df_raster %>%
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

df_fano_factor_final <- df_raster %>%
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

#options("browser"="brave")
#ggplotly(raster)
