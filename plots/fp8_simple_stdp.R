suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(stringr))
suppressPackageStartupMessages(library(purrr))

library(argparser)
include('plots/parse_inputs.R')

minifloat_values <- c(0., 0.00195312, 0.00390625, 0.00585938, 0.0078125, 0.00976562, 0.01171875, 0.01367188, 0.015625, 0.01757812, 0.01953125, 0.02148438, 0.0234375, 0.02539062, 0.02734375, 0.02929688, 0.03125, 0.03515625, 0.0390625, 0.04296875, 0.046875, 0.05078125, 0.0546875, 0.05859375, 0.0625, 0.0703125, 0.078125, 0.0859375, 0.09375, 0.1015625, 0.109375, 0.1171875, 0.125, 0.140625, 0.15625, 0.171875, 0.1875, 0.203125, 0.21875, 0.234375, 0.25, 0.28125, 0.3125, 0.34375, 0.375, 0.40625, 0.4375, 0.46875, 0.5, 0.5625, 0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375, 1., 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875, 2., 2.25, 2.5, 2.75, 3., 3.25, 3.5, 3.75, 4., 4.5, 5., 5.5, 6., 6.5, 7., 7.5, 8., 9., 10., 11., 12., 13., 14., 15., 16., 18., 20., 22., 24., 26., 28., 30., 32., 36., 40., 44., 48., 52., 56., 60., 64., 72., 80., 88., 96., 104., 112., 120., 128., 144., 160., 176., 192., 208., 224., 240., 256., 288., 320., 352., 384., 416., 448., 480., 0., -0.00195312, -0.00390625, -0.00585938, -0.0078125, -0.00976562, -0.01171875, -0.01367188, -0.015625, -0.01757812, -0.01953125, -0.02148438, -0.0234375, -0.02539062, -0.02734375, -0.02929688, -0.03125, -0.03515625, -0.0390625, -0.04296875, -0.046875, -0.05078125, -0.0546875, -0.05859375, -0.0625, -0.0703125, -0.078125, -0.0859375, -0.09375, -0.1015625, -0.109375, -0.1171875, -0.125, -0.140625, -0.15625, -0.171875, -0.1875, -0.203125, -0.21875, -0.234375, -0.25, -0.28125, -0.3125, -0.34375, -0.375, -0.40625, -0.4375, -0.46875, -0.5, -0.5625, -0.625, -0.6875, -0.75, -0.8125, -0.875, -0.9375, -1., -1.125, -1.25, -1.375, -1.5, -1.625, -1.75, -1.875, -2., -2.25, -2.5, -2.75, -3., -3.25, -3.5, -3.75, -4., -4.5, -5., -5.5, -6., -6.5, -7., -7.5, -8., -9., -10., -11., -12., -13., -14., -15., -16., -18., -20., -22., -24., -26., -28., -30., -32., -36., -40., -44., -48., -52., -56., -60., -64., -72., -80., -88., -96., -104., -112., -120., -128., -144., -160., -176., -192., -208., -224., -240., -256., -288., -320., -352., -384., -416., -448., -480.)
integer_values <- seq(0, 255)

lookup_table <- data.frame(
  original_value = integer_values,
  converted_value = minifloat_values
)
get_converted_value <- function(original_value) {
  if (original_value %in% lookup_table$original_value) {
    return(lookup_table$converted_value[lookup_table$original_value == original_value])
  } else {
    return(NA)  # or handle unmatched values as you see fit
  }
}

df_weights <- read.csv(file.path(argv$source, 'synapse_vars.csv'))
df_vars <- read.csv(file.path(argv$source, 'state_vars.csv'))

w_trace <- df_weights %>%
    filter(!str_detect(monitor, "ref"), variable=='w_plast')
w_trace$value <- map_dbl(w_trace$value, get_converted_value)

state_vars <- df_vars %>%
    filter(!str_detect(monitor, "ref"), id==0) # TODO 0?
state_vars$value <- map_dbl(state_vars$value, get_converted_value)

w_plot <- w_trace %>%
    ggplot(aes(x=time_ms, y=value, color=monitor)) +
    geom_line() + labs(x='time (ms)', y='weight (mV)', color='STDP method')

states <- state_vars %>%
    ggplot(aes(x=time_ms, y=value, color=variable)) +
    facet_grid(rows=vars(monitor), scales="free") +
    geom_line() + labs(x='time (ms)', y='state', color='STDP method')

print(argv$dest)
ggsave(argv$dest, states)
