suppressPackageStartupMessages(library(purrr))
suppressPackageStartupMessages(library(argparser))
suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(arrow))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(stringr))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(patchwork))
suppressPackageStartupMessages(library(plotly))

include('plots/parse_inputs.R')

library(wesanderson)
color_map <- wes_palette('GrandBudapest1')

wd = getwd()
dir_list <- Sys.glob(file.path(wd, argv$source, '*/', 'events_spikes.feather'))
events_list <- map(dir_list, read_feather)
dir_info <- Sys.glob(file.path(wd, argv$source, '*/', 'metadata.json'))
metadata_list <- map(dir_info, fromJSON)

# Arbitrary simulation result to use as control
arbitrary_control <- 3
control_dir <- Sys.glob(str_replace(dir_list[arbitrary_control],
                                    "events_spikes.feather",
                                    "spikes_*.csv"))
spike_origin <- map_chr(control_dir,
                        function(x) str_extract(x, "spikes_[:alpha:]+(?=.)"))
df_control <- map(control_dir, read.csv)
df_control <- map2(df_control, spike_origin, function(x,y) mutate(x, origin=y))
df_control <- list_rbind(df_control)
df_control <- df_control %>%
    group_by(time_ms, origin) %>%
    summarise(num_fetch = n())
df_control <- df_control %>%
    mutate(num_fetch=if_else(origin=="spikes_post",
                             1000*num_fetch,
                             num_fetch)) %>%
    group_by(time_ms) %>%
    summarize(num_fetch=sum(num_fetch))
df_control$group <- "control"

df_events <- map2(events_list, metadata_list,
                  function(x, y) mutate(x, group = y$event_condition))
df_events <- list_rbind(df_events)
df_events <- full_join(df_events, df_control)

t_init <- 0
t_final <- 50
fetch_dist_init <- df_events %>%
    filter(time_ms > t_init & time_ms < t_final) %>%
    ggplot(aes(x=group, y=num_fetch, fill=group)) +
    geom_violin() + theme(legend.position="none") + theme_bw()
trace_init <- df_events %>%
    filter(time_ms > t_init & time_ms < t_final) %>%
    ggplot(aes(x=time_ms, y=num_fetch, color=group)) + geom_line() +
    theme(legend.position="none") + theme_bw()

t_init <- 99500
t_final <- 100000
fetch_dist_final <- df_events %>%
    filter(time_ms > t_init & time_ms < t_final) %>%
    ggplot(aes(x=group, y=num_fetch, fill=group)) +
    geom_violin() + theme(legend.position="none") + theme_bw()
trace_final <- df_events %>%
    filter(time_ms > t_init & time_ms < t_final) %>%
    ggplot(aes(x=time_ms, y=num_fetch, color=group)) + geom_line() +
    theme_bw()

df_spk <- read.csv(str_replace(nth(dir_list, -1), "events_spikes.feather", "spikes_post.csv"))
rates <- df_spk %>% ggplot(aes(x=time_ms)) + geom_histogram()

n_fetch <- (trace_init + trace_final) /
       (fetch_dist_init + fetch_dist_final) /
       rates + plot_annotation(tag_levels='A')
ggsave(argv$dest, n_fetch)
