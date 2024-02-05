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

read_folder <- function(dir_path) {
    event_data <- read_feather(file.path(dir_path, 'events_spikes.feather'))
    metadata_info <- fromJSON(file.path(dir_path, 'metadata.json'))
    control_paths <- Sys.glob(file.path(dir_path, 'spikes_*.csv'))

    spike_origin <- map_chr(control_paths,
                            function(x) str_extract(x, "spikes_[:alpha:]+(?=.)"))
    df_control <- map(control_paths, read.csv)
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
    df_control$N_post <- metadata_info$N_post

    df_events <- event_data %>%
        mutate(group = metadata_info$event_condition,
               N_post = metadata_info$N_post)
    df_events <- full_join(df_events, df_control)

    return(df_events)
}

dir_list <- Sys.glob(file.path(wd, argv$source, '*'))
df_events <- map(dir_list, read_folder)
df_events <- list_rbind(df_events)

t_init <- 2000
t_final <- 2250
df_events <- df_events %>%
    mutate(group = str_extract(group, '[:digit:]\\.*[:digit:]*|control')) %>%
    filter(time_ms > t_init & time_ms < t_final) %>%
    group_by(N_post, group) %>% mutate(mean_fetch=mean(num_fetch))
multi_fetch_avg <- df_events %>%
    ggplot(aes(x=group, y=num_fetch)) +
    geom_boxplot(aes(fill=group)) + theme_bw() +
    theme(legend.position="none", text=element_text(size=16)) +
    geom_point(aes(y=mean_fetch, color=group), shape=2, size=7) +
    facet_wrap(~N_post) + scale_y_log10() +
    labs(x=element_blank(), y='# Memory access') +
    scale_fill_manual(values=color_map[c(2, 4)]) +
    scale_color_manual(values=color_map[c(2, 4)])

ggsave(argv$dest, multi_fetch_avg)
