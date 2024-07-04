suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(stringr))
suppressPackageStartupMessages(library(purrr))
suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(patchwork))
suppressPackageStartupMessages(library(forcats))
suppressPackageStartupMessages(library(wesanderson))

library(argparser)
include('plots/parse_inputs.R')

minifloat_values <- c(0., 0.00195312, 0.00390625, 0.00585938, 0.0078125, 0.00976562, 0.01171875, 0.01367188, 0.015625, 0.01757812, 0.01953125, 0.02148438, 0.0234375, 0.02539062, 0.02734375, 0.02929688, 0.03125, 0.03515625, 0.0390625, 0.04296875, 0.046875, 0.05078125, 0.0546875, 0.05859375, 0.0625, 0.0703125, 0.078125, 0.0859375, 0.09375, 0.1015625, 0.109375, 0.1171875, 0.125, 0.140625, 0.15625, 0.171875, 0.1875, 0.203125, 0.21875, 0.234375, 0.25, 0.28125, 0.3125, 0.34375, 0.375, 0.40625, 0.4375, 0.46875, 0.5, 0.5625, 0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375, 1., 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875, 2., 2.25, 2.5, 2.75, 3., 3.25, 3.5, 3.75, 4., 4.5, 5., 5.5, 6., 6.5, 7., 7.5, 8., 9., 10., 11., 12., 13., 14., 15., 16., 18., 20., 22., 24., 26., 28., 30., 32., 36., 40., 44., 48., 52., 56., 60., 64., 72., 80., 88., 96., 104., 112., 120., 128., 144., 160., 176., 192., 208., 224., 240., 256., 288., 320., 352., 384., 416., 448., 480., 0., -0.00195312, -0.00390625, -0.00585938, -0.0078125, -0.00976562, -0.01171875, -0.01367188, -0.015625, -0.01757812, -0.01953125, -0.02148438, -0.0234375, -0.02539062, -0.02734375, -0.02929688, -0.03125, -0.03515625, -0.0390625, -0.04296875, -0.046875, -0.05078125, -0.0546875, -0.05859375, -0.0625, -0.0703125, -0.078125, -0.0859375, -0.09375, -0.1015625, -0.109375, -0.1171875, -0.125, -0.140625, -0.15625, -0.171875, -0.1875, -0.203125, -0.21875, -0.234375, -0.25, -0.28125, -0.3125, -0.34375, -0.375, -0.40625, -0.4375, -0.46875, -0.5, -0.5625, -0.625, -0.6875, -0.75, -0.8125, -0.875, -0.9375, -1., -1.125, -1.25, -1.375, -1.5, -1.625, -1.75, -1.875, -2., -2.25, -2.5, -2.75, -3., -3.25, -3.5, -3.75, -4., -4.5, -5., -5.5, -6., -6.5, -7., -7.5, -8., -9., -10., -11., -12., -13., -14., -15., -16., -18., -20., -22., -24., -26., -28., -30., -32., -36., -40., -44., -48., -52., -56., -60., -64., -72., -80., -88., -96., -104., -112., -120., -128., -144., -160., -176., -192., -208., -224., -240., -256., -288., -320., -352., -384., -416., -448., -480.)
integer_values <- seq(0, 255)

lookup_table <- data.frame(
  original_value = integer_values,
  converted_value = minifloat_values
)
minifloat2decimal <- function(original_value) {
  if (original_value %in% lookup_table$original_value) {
    return(lookup_table$converted_value[lookup_table$original_value == original_value])
  } else {
    return(NA)  # or handle unmatched values as you see fit
  }
}

color_map <- wes_palette('FantasticFox1')

wd = getwd()
data_path <- Sys.glob(file.path(wd, argv$source, "*"))
data_path <- data_path[str_detect(data_path, ".png", negate=T)]
metadata <- map(file.path(data_path, "metadata.json"), fromJSON)
stoch_round <- map_lgl(metadata, \(x) x$stochastic_rounding)
small_w <- map_dbl(metadata, \(x) x$w_init) == 0.02539062

selected_folder <- stoch_round & !small_w
sim_id <- match(T, selected_folder)
df_vars <- read.csv(file.path(data_path[sim_id], 'state_vars.csv'))
state_vars <- df_vars %>%
    filter(!str_detect(monitor, "ref"), id==0) %>%
    mutate(monitor = str_replace(monitor,
                                 "statemon_pre_neurons",
                                 "presynaptic")) %>%
    mutate(monitor = str_replace(monitor,
                                 "statemon_post_neurons",
                                 "postsynaptic")) %>%
    mutate(variable = str_replace(variable, "Ca", "x")) %>%
    mutate(variable = str_replace(variable, "g", "PSP"))
state_vars$value <- map_dbl(state_vars$value, minifloat2decimal)
states_stoch <- state_vars %>%
    ggplot(aes(x=time_ms, y=value, color=variable)) + theme_bw() +
    facet_grid(rows=vars(fct_rev(monitor)), scales="free") +
    geom_line() + labs(x='time (ms)', y='magnitude (a.u.)', color='state variables') +
    xlim(c(363, 713)) + scale_color_manual(values=color_map[c(1, 3, 5)]) +
    guides(color=guide_legend(override.aes=list(linewidth=5))) +
    theme(legend.position="none", text = element_text(size=16))

df_weights <- read.csv(file.path(data_path[sim_id], 'synapse_vars.csv'))
w_trace <- df_weights %>%
    filter(!str_detect(monitor, "ref"), variable=='w_plast')
w_trace$value <- map_dbl(w_trace$value, minifloat2decimal)
temp_weights <- w_trace %>%
    filter(id==0 | id==1 | id==2 | id==3 | id==4 | id==5 | id==6 | id==7) %>%
    ggplot(aes(x=time_ms, y=value, group=id)) +
    geom_line(alpha=.6, color=color_map[3])
avg_weight <- w_trace %>%
    group_by(time_ms) %>%
    mutate(value = mean(value))
weights_stoch_high <- temp_weights +
    geom_line(data=avg_weight, aes(x=time_ms, y=value),
              color=color_map[4], linewidth=1) +
    labs(x='time (ms)', y='weight (a.u.)') +
    xlim(c(363, 1957)) + theme_bw() + theme(text=element_text(size=16))

selected_folder <- stoch_round & small_w
sim_id <- match(T, selected_folder)
df_weights <- read.csv(file.path(data_path[sim_id], 'synapse_vars.csv'))
w_trace <- df_weights %>%
    filter(!str_detect(monitor, "ref"), variable=='w_plast')
w_trace$value <- map_dbl(w_trace$value, minifloat2decimal)
temp_weights <- w_trace %>%
    filter(id==0 | id==1 | id==2 | id==3 | id==4 | id==5 | id==6 | id==7) %>%
    ggplot(aes(x=time_ms, y=value, group=id)) +
    geom_line(alpha=.6, color=color_map[3])
avg_weight <- w_trace %>%
    group_by(time_ms) %>%
    mutate(value = mean(value))
weights_stoch_low <- temp_weights +
    geom_line(data=avg_weight, aes(x=time_ms, y=value),
              color=color_map[4], linewidth=1) +
    labs(x='time (ms)', y='weight (a.u.)') +
    xlim(c(363, 1957)) + theme_bw() + theme(text=element_text(size=16))

selected_folder <- !stoch_round & !small_w
sim_id <- match(T, selected_folder)

df_vars <- read.csv(file.path(data_path[sim_id], 'state_vars.csv'))
state_vars <- df_vars %>%
    filter(!str_detect(monitor, "ref"), id==0) %>%
    mutate(monitor = str_replace(monitor,
                                 "statemon_pre_neurons",
                                 "presynaptic")) %>%
    mutate(monitor = str_replace(monitor,
                                 "statemon_post_neurons",
                                 "postsynaptic")) %>%
    mutate(variable = str_replace(variable, "Ca", "x")) %>%
    mutate(variable = str_replace(variable, "g", "PSP"))
state_vars$value <- map_dbl(state_vars$value, minifloat2decimal)
states_det <- state_vars %>%
    ggplot(aes(x=time_ms, y=value, color=variable)) + theme_bw() +
    facet_grid(rows=vars(fct_rev(monitor)), scales="free") +
    geom_line() + labs(x='time (ms)', y='magnitude (a.u.)', color='state variables') +
    xlim(c(363, 713)) + scale_color_manual(values=color_map[c(1, 3, 5)]) +
    guides(color=guide_legend(override.aes=list(linewidth=5))) +
    theme(text=element_text(size=16))

df_weights <- read.csv(file.path(data_path[sim_id], 'synapse_vars.csv'))
w_trace <- df_weights %>%
    filter(!str_detect(monitor, "ref"), variable=='w_plast')
w_trace$value <- map_dbl(w_trace$value, minifloat2decimal)
weights_det_high <- w_trace %>%
    filter(id==0) %>%
    ggplot(aes(x=time_ms, y=value, group=id)) +
    geom_line(alpha=.6, color=color_map[3]) + theme_bw() +
    theme(text=element_text(size=16)) +
    scale_color_manual(values = color_map[3]) +
    labs(x='time (ms)', y='weight (a.u.)') +
    xlim(c(363, 1957))

selected_folder <- !stoch_round & small_w
sim_id <- match(T, selected_folder)

df_weights <- read.csv(file.path(data_path[sim_id], 'synapse_vars.csv'))
w_trace <- df_weights %>%
    filter(!str_detect(monitor, "ref"), variable=='w_plast')
w_trace$value <- map_dbl(w_trace$value, minifloat2decimal)
weights_det_low <- w_trace %>%
    filter(id==0) %>%
    ggplot(aes(x=time_ms, y=value, group=id)) +
    geom_line(alpha=.6, color=color_map[3]) + theme_bw() +
    theme(text=element_text(size=16)) +
    scale_color_manual(values = color_map[3]) +
    labs(x='time (ms)', y='weight (a.u.)') +
    xlim(c(363, 1957))

fig_states <- states_det / states_stoch + plot_annotation(tag_levels='A')
fig_weights <- weights_det_low + weights_stoch_low +
    weights_det_high + weights_stoch_high + plot_annotation(tag_levels='A')

ggsave(str_replace(argv$dest, '.png', '_states.png'), fig_states)
ggsave(str_replace(argv$dest, '.png', '_weights.png'), fig_weights)
