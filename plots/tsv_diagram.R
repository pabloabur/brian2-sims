suppressPackageStartupMessages(library(magick))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(patchwork))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(purrr))

library(argparser)
include('plots/parse_inputs.R')

wd = getwd()
data_path <- Sys.glob(file.path(wd, argv$source, '*'))

panes <- map(data_path, ~image_read(.x) %>%
    image_ggplot() + theme(text = element_text(size=16),
                           plot.margin = margin(2, 2, 2, 2, "mm")))

fig <- panes[[1]] + panes[[2]] + panes[[3]] + panes[[4]] + plot_annotation(tag_levels='A')
ggsave(argv$dest, fig)
