library(arrow)
library(visNetwork)
links <- arrow::read_feather('~/git/brian2-sims/links.feather')
nodes <- arrow::read_feather('~/git/brian2-sims/nodes.feather')
network <- graph_from_data_frame(d=links, vertices = nodes)

types <- c('red', 'blue')
my_color <- types[as.numeric(as.factor(V(network)$type))]
V(network)$color <- my_color
visIgraph(network)