library(dplyr)
library(jsonlite)
library(purrr)
library(ggplot2)
library(patchwork)
library(forcats)
library(wesanderson)

library(argparser)
include('plots/parse_inputs.R')

wd = getwd()
data_path <- file.path(wd, argv$source)

metadata <- fromJSON(file.path(data_path, 'metadata.json'))

pal <- wes_palette('Moonrise1', n=4)
color_map <- c('int4'=pal[1], 'int8'=pal[2], 'fp8'=pal[4], 'fp64'=pal[3])

rates <- arrow::read_feather(file.path(data_path, 'rates.feather'))
p1 <- ggplot(rates, aes(x=times_ms)) +
          geom_line(aes(y=rate_Hz, color=resolution)) +
          labs(tag='A', x='time (ms)', y='frequency (Hz)', color='Bit-precision') +
          scale_color_manual(values=color_map) + theme_bw() +
          ylim(c(0, 250)) + scale_x_continuous(expand=c(0,0)) +
          theme(panel.grid.minor=element_blank(),
                panel.grid.major.y=element_blank()) +
          guides(color=guide_legend(override.aes=list(size=4))) +
          annotate(geom='text', x=200, y=230, label='8 Hz') +
          annotate(geom='text', x=1200, y=230, label='6 Hz') +
          annotate(geom='text', x=2200, y=230, label='4 Hz') +
          annotate(geom='text', x=3200, y=230, label='2 Hz') +
          annotate(geom='text', x=4200, y=230, label='1 Hz')

traces <- arrow::read_feather(file.path(data_path, 'traces.feather'))
traces <- traces %>%
    filter(id==0)

time_ref0 <- 500
time_ref1 <- 25

create_plot1 <- function(traces, t){
    new_strip_label <- as_labeller(c(`gtot`='PSP', `vm`='Vm'))
    p <- traces %>%
        filter(time_ms>=t+time_ref0 & time_ms<=t+time_ref0+time_ref1) %>%
        ggplot(aes(x=time_ms)) +
            geom_line(aes(y=values, color=resolution)) +
            facet_grid(rows=vars(type), labeller=new_strip_label) +
            theme_bw() + theme(panel.grid.minor=element_blank(),
                               panel.grid.major=element_blank(),
                               legend.position='none',
                               strip.background=element_blank(),
                               strip.text.y=element_blank(),
                               axis.text.x=element_blank(),
                               axis.ticks.x=element_blank()) +
            labs(x=element_blank(), y=element_blank()) +
            scale_y_continuous(breaks=c(0, .5, 1)) +
            scale_color_manual(values=color_map)

    return(p)
}

t_refs <- metadata$stim_time_tag

p2 <- map(t_refs, ~create_plot1(traces, .))
p2[[1]] <- p2[[1]] + labs(tag='B')
p2[[5]] <- p2[[5]] + theme(strip.text.y=element_text(angle=-90))

corr <- arrow::read_feather(file.path(data_path, 'corr.feather'))
p3 <- corr %>%
    group_by(input_rate, pair) %>%
    ggplot(aes(pair, coef, fill=pair)) + geom_boxplot() +
    facet_grid(cols=vars(fct_rev(input_rate))) +
    theme_bw() +
    theme(legend.position='none',
          strip.background=element_blank(),
          strip.text.x=element_blank(),
          panel.grid.minor=element_blank(),
          axis.ticks.x=element_blank(),
          axis.text.x=element_blank()) +
    scale_fill_manual(values=color_map) +
    labs(x=element_blank(), y='correlation', tag='C')

cch <- arrow::read_feather(file.path(data_path, 'cch.feather'))
p4 <- cch %>%
    ggplot(aes(x=lags, color=resolution)) + geom_line(aes(y=cch)) +
    # facet_grid did not work with free_y here
    facet_wrap(vars(fct_rev(as.factor(input_rate))), scales='free_y', nrow=1) +
    coord_cartesian(xlim=c(-5, 5)) + theme_bw() +
    theme(legend.position='none',
          panel.grid.minor=element_blank(),
          strip.background=element_blank(),
          strip.text.x=element_blank()) +
    guides(x=guide_axis(angle=60)) +
    scale_color_manual(values=color_map) +
    labs(x='lags (ms)', y='correlation', tag='D')

fig <- ((p1 / (p2[[1]] | p2[[2]] | p2[[3]] | p2[[4]] | p2[[5]])) / p3) / p4
ggsave(argv$dest, fig)
