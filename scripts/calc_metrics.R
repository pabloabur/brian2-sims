library(dplyr)
library(ggplot2)
library(purrr)
library(tidyr)
library(arrow)

# Below to process mus silicium
# spikes = read.csv('spikes.csv')
# 
# speaker <- spikes[, 1]
# digit <- spikes[, 2]
# df<-data.frame(speaker, digit)
# df$spike_times <- unname(spikes[, -1:-2])
# df<-df %>%
#     group_by(speaker, digit) %>%
#     mutate(trial=row_number())
# 
# ggplot(df[df$speaker==1 & df$digit==0, ][1, ], aes(x=spike_times, y=seq(1, 40)))
#     + geom_point()


# leading spikes with feather
df2 <- arrow::read_feather('output_spikes.feather')
df4 <- arrow::read_feather('events_spikes.feather')
df4 <- df4 %>%
  mutate(t_start_probe=tstop_ms-10, t_stop_probe=tstop_ms+10)

hits <- pmap(df4, ~ df2 %>%
                          filter(between(`time_ms`, ..4, ..5)) %>%
                          summarise(id))
df5 <- df4 %>% mutate(unique_hits=map(hits, unique))

# Plotting
temp_tab <- df5 %>% select(label, unique_hits) %>% unnest_longer(unique_hits)
conf_mat <- data.frame(label=temp_tab$label, pred=temp_tab$unique_hits$id)
heatmap(table(conf_mat))
table(conf_mat)
# In case we want to check nonresponsive cases any(unlist(df3$test[115]))