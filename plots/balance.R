library(purrr)
library(stringr)
library(arrow)
library(ggplot2)
library(dplyr)
library(patchwork)
library(wesanderson)

library(argparser)
include('plots/parse_inputs.R')

wd = getwd()
dir_list <- Sys.glob(file.path(wd, argv$source, '*/'))

vm <- map(file.path(dir_list, 'voltages.feather'), read_feather)
rates <- map(file.path(dir_list, 'avg_rates.feather'), read_feather)
weights <- map(file.path(dir_list, 'weights.feather'), read_feather)

sim_names <- map_chr(str_split(dir_list, '\\/'), ~nth(., -2))
vm <- set_names(vm, sim_names)
separated_rates <- set_names(rates, sim_names)
weights <- set_names(weights, sim_names)

color_map <- wes_palette('Moonrise1')

rates <- reduce(rates, full_join)
rates_stats <- rates %>%
    group_by(inhibitory_weight, resolution) %>%
    summarise(mean_freq=mean(frequency_Hz), sd_freq=sd(frequency_Hz))
p1 <- rates_stats %>%
    ggplot(aes(x=inhibitory_weight, y=mean_freq)) +
    geom_point(color=color_map[[4]]) +
    geom_errorbar(aes(ymin=mean_freq-sd_freq, ymax=mean_freq+sd_freq)) +
    geom_line() +
    facet_wrap(vars(resolution), scales='free') +
    theme_bw() + theme(text = element_text(size=16)) +
    labs(x='mean inhibitory weight (a.u.)', y='frequency (Hz)')

# Chosen for convenience a not so strong inhibition
p2 <- ggplot(vm$'16-01_11h23m35s', aes(x=time_ms, y=values, color=resolution)) +
    geom_line() + xlim(0, 0.1) + ylim(-1, 1) +
    labs(x='time (s)', y='Vm', color='Bit-precision') +
    guides(color=guide_legend(override.aes=list(linewidth=4))) +
    theme_bw() + scale_color_manual(values=color_map[c(2, 4)]) +
    theme(text = element_text(size=16))

# Next plots were handpicked. If you dont have data, run it and choose file
weights$'17-01_12h50m07s' <- weights$'17-01_12h50m07s' %>%
    filter(resolution=='fp64' | resolution=='fp8')
weights$'17-01_12h50m07s'[weights$'17-01_12h50m07s'=='fp8'] = 'fp8*'
weights$'16-01_11h41m10s' <- weights$'16-01_11h41m10s' %>%
    filter(resolution=='fp8')
selected_weights <- full_join(weights$'17-01_12h50m07s', weights$'16-01_11h41m10s')
p3 <- selected_weights %>%
    slice_sample(n=1000) %>%
    ggplot(aes(values)) +
    geom_histogram(fill=color_map[[4]]) +
    facet_wrap(vars(resolution), scales='free') + 
    theme_bw() + labs(x='inhibitory weights (a.u.)') +
    theme(text = element_text(size=16)) +
    scale_x_continuous(breaks = scales::pretty_breaks(n = 4))

p4 <- selected_weights  %>%
    slice_sample(n=1000) %>%
    ggplot() +
    geom_qq(aes(sample=values), color=color_map[[2]]) + stat_qq_line(aes(sample=values)) +
    facet_wrap(vars(resolution), scales='free') +
    labs(x='theoretical quantiles', y='sampled quantiles') +
    theme_bw() + theme(text = element_text(size=16)) +
    scale_x_continuous(breaks = scales::pretty_breaks(n = 3))

fig <- (p2 / p3 / p4) + plot_annotation(tag_levels='A')
ggsave(str_replace(argv$dest, '.png', '1.png'), p1)
ggsave(str_replace(argv$dest, '.png', '2.png'), fig)

print('Average rate of a net running with uniform distribution:')
print(separated_rates$'2023.01.17_15.40' %>% filter(resolution=='fp8'))
