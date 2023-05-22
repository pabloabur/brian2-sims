library(ggplot2)
library(arrow)
library(plotly)

library(argparser)
include('plots/parse_inputs.R')


a<-read_feather('sim_data/maoi/spikes.feather')
b<-ggplot(a, aes(x=times, y=indices, color=type)) + geom_point() + theme_bw()
# ggplotly(b)