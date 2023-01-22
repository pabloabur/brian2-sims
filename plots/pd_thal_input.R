library(ggplot2)
library(tidyr)
library(stringr)
library(dplyr)
library(arrow)
library(purrr)
library(jsonlite)
library(wesanderson)
library(plotly)

args = commandArgs(trailingOnly=T)
if (length(args)==0){
    stop("Folder with data must be provided")
}else{
    folder = args[1]
}

color_map <- wes_palette('Moonrise2')

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
    mutate(type=if_else(str_sub(layer, -1, -1)=='e', 'Exc', 'Inh')) #%>%
    #filter(t>=min_time & t<=max_time)

raster <- ggplot(df_raster, aes(x=t, y=i, color=type)) +
         geom_point(shape=20, size=1) + theme_bw() +
         theme(panel.grid.minor=element_blank(),
               panel.grid.major=element_blank(),
               axis.text.y=element_blank(),
               axis.ticks.y=element_blank()) +
         guides(color=guide_legend(override.aes=list(size=5))) +
         facet_grid(rows=vars(laminae), scales="free") +
         labs(x='time (ms)', y=element_blank())

histograms <- df_raster %>%
    mutate(t=t%%1000) %>%
    ggplot(aes(x=t, color=type)) + geom_step(stat='bin', bins=50) +
    facet_grid(rows=vars(laminae), scales='free') + theme_bw() +
    scale_color_manual(values=color_map) +
    labs(x='time (ms)') + xlim(670, 730)

histograms
dev.new()
raster
tr<-read_feather(file.path(dir, 'traces.feather'))
ggplotly(ggplot(tr, aes(x=t, y=v)) + geom_line())
