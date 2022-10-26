library(arrow)
library(visNetwork)
library(igraph)

links <- arrow::read_feather('links.feather')
nodes <- arrow::read_feather('nodes.feather')
network <- graph_from_data_frame(d=links, vertices = nodes)

types <- c('red', 'blue')
my_color <- types[as.numeric(as.factor(V(network)$type))]
V(network)$color <- my_color
visIgraph(network)
