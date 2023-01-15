library(purrr)
library(arrow)
library(ggplot2)
library(dplyr)

args = commandArgs(trailingOnly=T)
if (length(args)==0){
    stop("Folder with data folders must be provided")
}else{
    folder = args[1]
}

wd = getwd()
dir_list <- Sys.glob(file.path(wd, folder, '2023*'))

vm <- map(file.path(dir_list, 'voltages.feather'), read_feather)
rates <- map(file.path(dir_list, 'avg_rates.feather'), read_feather)
weights <- map(file.path(dir_list, 'weights.feather'), read_feather)

rates <- reduce(rates, full_join)
p1 <- rates %>%
    ggplot(aes(x=inhibitory_weight, y=frequency_Hz)) +
    geom_point() + geom_smooth(span=0.8) + facet_wrap(vars(resolution), scales='free')

# Chosen for convenience
p2 <- ggplot(vm[[2]], aes(x=time_ms, y=values, color=resolution)) +
    geom_line() + xlim(0, 0.1) + ylim(-4, 1)

p3 <- ggplot(weights[[2]], aes(values)) +
    geom_histogram() + facet_wrap(vars(resolution), scales='free')

p4 <- weights[[3]] %>%
    slice_sample(n=1000) %>%
    ggplot() +
    geom_qq(aes(sample=values)) + stat_qq_line(aes(sample=values)) +
    facet_wrap(vars(resolution), scales='free')
# TODO fig fp8 qqline for high and low variance
