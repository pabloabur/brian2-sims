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

#argv$source <- 'sim_data/stdp_test/'

wd = getwd()
dir_list <- Sys.glob(file.path(wd, argv$source, '*/', 'events_spikes.feather'))
events_list <- map(dir_list, read_feather)
dir_info <- Sys.glob(file.path(wd, argv$source, '*/', 'metadata.json'))
metadata_list <- map(dir_info, fromJSON)

# Arbitrary simulation result to use as control, i.e. folder 1
control_dir <- str_replace(dir_list[1], "events_spikes.feather", "spikes_pre.csv")
df_control <- read.csv(control_dir)
df_control$group <- "control"

df_events <- map2(events_list, metadata_list,
                  function(x, y) mutate(x, group = y$event_condition))
rm(events_list)
df_events <- list_rbind(df_events)
df_events <- full_join(df_events, df_control)
rm(df_control)
gc()
p <- df_events %>%
    group_by(time_ms, group) %>%
    summarise(num_fetch=n()) %>%
    ggplot(aes(x=time_ms, y=num_fetch, color=group)) + geom_line()

df_weights <- read.csv('sim_data/stdp_test/02-10_18h40m08s//synapse_vars_weights.csv')
w_distribution <- df_weights %>%
    filter(label=='Original') %>%
    ggplot(aes(x=w_plast, color=label, fill=label)) + geom_histogram(alpha=0.8) +
    theme_bw() + theme(legend.position="none") +
    labs(x='weights (mV)', color=element_blank(), fill=element_blank())
df_spk <- read.csv('sim_data/stdp_test/02-10_18h40m08s//spikes_post.csv')
rates <- df_spk %>% ggplot(aes(x=time_ms)) + geom_histogram()
fig <- p / w_distribution / rates
ggsave(argv$dest, fig)
