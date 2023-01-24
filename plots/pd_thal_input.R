library(ggplot2)
library(tidyr)
library(stringr)
library(dplyr)
library(arrow)
library(purrr)
library(jsonlite)
library(wesanderson)
library(plotly)
library(patchwork)

args = commandArgs(trailingOnly=T)
if (length(args)==0){
    stop("Folder with data must be provided")
}else{
    folder = args[1]
    save_path <- args[2]  # with name and extension
}

color_map <- wes_palette('Moonrise1')[c(4, 2)]

wd = getwd()
dir <- Sys.glob(file.path(wd, folder))

spikes <- read_feather(file.path(dir, "spikes.feather"))
metadata <- fromJSON(file.path(dir, "metadata.json"))

tsim <- as.double(str_sub(metadata$duration, 1, -2))

# Units considered in ms. Input comes around 700ms after each trial
trials <- seq(tsim)
min_time <- trials[1]*1000 - 330
max_time <- trials[1]*1000 - 270
p_sample <- 0.025
df_raster <- spikes %>%
    group_by(layer) %>%
    slice_sample(prop=p_sample) %>%
    unnest_longer(t) %>%
    mutate(laminae=str_sub(layer, 1, -2)) %>%
    mutate(type=if_else(str_sub(layer, -1, -1)=='e', 'Exc', 'Inh'))

raster <- df_raster %>%
         filter(t>=min_time & t<=max_time) %>%
         ggplot(aes(x=t, y=i, color=type)) +
         geom_point(shape=20, size=1) + theme_bw() +
         theme(panel.grid.minor=element_blank(),
               panel.grid.major=element_blank(),
               axis.text.y=element_blank(),
               axis.ticks.y=element_blank()) +
         guides(color=guide_legend(override.aes=list(size=5))) +
         scale_color_manual(values=color_map) +
         facet_grid(rows=vars(laminae), scales="free") +
         labs(x='time (ms)', y=element_blank())

histograms <- df_raster %>%
    mutate(t=t%%1000) %>%
    ggplot(aes(x=t, color=type)) + geom_step(stat='bin', bins=50) +
    guides(color=guide_legend(override.aes=list(size=4))) +
    facet_grid(rows=vars(laminae), scales='free') + theme_bw() +
    theme(legend.position='none') +
    scale_color_manual(values=color_map) +
    labs(x='time (ms)') + xlim(670, 730)

fig <- (raster + histograms) + plot_annotation(tag_levels='A')
ggsave(save_path, fig)
