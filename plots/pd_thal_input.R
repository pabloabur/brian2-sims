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
suppressPackageStartupMessages(library(magick))

library(argparser)
include('plots/parse_inputs.R')

color_map <- wes_palette('Moonrise1')[c(4, 2)]

wd = getwd()
dir <- Sys.glob(file.path(wd, argv$source))

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
         labs(x='time (ms)', y=element_blank()) +
         theme(text = element_text(size=16)) +
         scale_x_continuous(breaks = scales::pretty_breaks(n = 4))

binned_data <- df_raster %>%
    mutate(t=t%%1000) %>%
    mutate(bin = cut(t,
                     breaks=seq(min(t), max(t), by=1),
                     include.lowest=TRUE,
                     right=FALSE)) %>%
    filter(t>=690 & t<=734) %>%
    group_by(layer, bin) %>%
    mutate(count = n()/tsim)  %>%
    mutate(proxy_layer=case_when(laminae=='L23' ~ 30,
                                 laminae=='L4' ~ 18,
                                 laminae=='L5' ~ 13,
                                 laminae=='L6' ~ 5)) %>%
    mutate(adjusted_count = count + proxy_layer)

orig_like <- binned_data %>%
    ggplot(aes(x=t, y=adjusted_count, group=layer, fill=type, color=type)) +
    geom_step(size=1.5) +
    theme_bw() +
    theme(text = element_text(size=16),
          legend.position="none",
          panel.border=element_blank(),
          axis.line.x = element_line(color='black'),
          panel.grid.minor=element_blank(),
          panel.grid.major=element_blank(),
          axis.text.y=element_blank()) +
    scale_color_manual(values=c("Exc"="black", "Inh"="grey")) +
    scale_x_continuous(name='time (ms)',
                       labels=as.character(seq(-10, 30, by=10)),
                       breaks=seq(690, 730, by=10)) +
    scale_y_continuous(name=NULL, breaks=NULL) +
    annotate(geom='text', x=695, y=34, label='L2/3', size=5) +
    annotate(geom='text', x=695, y=23, label='L4', size=5) +
    annotate(geom='text', x=695, y=16, label='L5', size=5) +
    annotate(geom='text', x=695, y=8, label='L6', size=5) +
    geom_segment(x=723, xend=723, y=32, yend=42, color='black', linewidth=1.5) +
    annotate(geom='text', x=730, y=36, label='10\nspikes', size=5)

orig_PD_thal <- image_read("sim_data/ch2/pd_thal/orig_pd_thal.png")
orig_PD_thal <- orig_PD_thal %>%
    image_ggplot() + theme(text = element_text(size=16))

histograms <- df_raster %>%
    mutate(t=t%%1000) %>%
    group_by(t, i) %>%
    ggplot(aes(x=t, color=type)) + geom_step(stat='bin', bins=50) +
    guides(color=guide_legend(override.aes=list(size=4))) +
    facet_grid(rows=vars(laminae), scales='free') + theme_bw() +
    theme(legend.position='none') +
    scale_color_manual(values=color_map) +
    labs(x='time (ms)') + xlim(670, 730) +
    theme(text = element_text(size=16))

fig <- (raster + histograms) + plot_annotation(tag_levels='A')
fig_comp <- orig_PD_thal / orig_like + plot_annotation(tag_levels='A')

ggsave(argv$dest, fig)
ggsave(str_replace(argv$dest, '.png', '_comp.png'), fig_comp)
