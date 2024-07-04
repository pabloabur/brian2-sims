suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(stringr))
suppressPackageStartupMessages(library(purrr))
suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(patchwork))
suppressPackageStartupMessages(library(forcats))
suppressPackageStartupMessages(library(wesanderson))
suppressPackageStartupMessages(library(latex2exp))

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

generate_plot <- function(df, time) {
  df <- df %>%
      filter(time_ms == time &
             variable=='w_plast' &
             monitor=='statemon_post_synapse') %>%
      mutate(id = id - 29.5)
  df$value <- map_dbl(df$value, minifloat2decimal)
  kernel <- df %>%
      ggplot(aes(x=id, y=value)) + geom_line(color=color_map[3]) + theme_bw() +
      labs(x=TeX(r'($\Delta$\;t (ms))'), y=element_blank(), color=element_blank()) +
      guides(color=guide_legend(override.aes=list(linewidth=4))) +
      theme(legend.position = c(0.25, 0.8), text=element_text(size=16))
  return(kernel)
}

color_map <- wes_palette('FantasticFox1')

wd = getwd()
data_path <- Sys.glob(file.path(wd, argv$source, "*"))
data_path <- data_path[str_detect(data_path, ".png", negate=T)]
metadata <- map(file.path(data_path, "metadata.json"), fromJSON)

small_w_exp <- map_int(metadata, \(x) x$w_init) == 16

df_weights_high <- read.csv(file.path(data_path[!small_w_exp], 'synapse_vars.csv'))
kernel_high_middle <- generate_plot(df_weights_high, 1800)
kernel_high_middle <- kernel_high_middle + labs(y='weight (a.u.)')
kernel_high_final <- generate_plot(df_weights_high, 9000)

df_weights_small <- read.csv(file.path(data_path[small_w_exp], 'synapse_vars.csv'))
kernel_small_middle <- generate_plot(df_weights_small, 1800)
kernel_small_middle <- kernel_small_middle + labs(y='weight (a.u.)')
kernel_small_final <- generate_plot(df_weights_small, 9000)

fig_kernel <- wrap_elements(kernel_high_middle + kernel_high_final +
                            plot_annotation(title='A', theme=theme(text=element_text(size=16)))) /
              wrap_elements(kernel_small_middle + kernel_small_final +
                            plot_annotation(title='B', theme=theme(text=element_text(size=16))))

ggsave(argv$dest, fig_kernel)
