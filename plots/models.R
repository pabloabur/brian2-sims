library(dplyr)
library(jsonlite)
library(purrr)
library(ggplot2)
library(patchwork)
library(forcats)
library(wesanderson)

args = commandArgs(trailingOnly=T)
if (length(args)==0){
    stop("Folder with data folders must be provided")
}else{
    folder = args[1]
}

wd = getwd()
data_path <- file.path(wd, folder)

metadata <- fromJSON(file.path(data_path, 'metadata.json'))

pal <- wes_palette('Moonrise1', n=4)
color_map <- c('int4'=pal[1], 'int8'=pal[2], 'fp8'=pal[3], 'fp64'=pal[4])

rates <- arrow::read_feather(file.path(data_path, 'rates.feather'))
p1 <- ggplot(rates, aes(x=times_ms)) +
          geom_line(aes(y=rate_Hz, color=resolution)) +
          labs(tag='A') + scale_color_manual(values=color_map)

traces <- arrow::read_feather(file.path(data_path, 'traces.feather'))
traces <- traces %>%
    filter(id==0)

time_ref0 <- 500
time_ref1 <- 25

create_plot1 <- function(traces, t){
    new_strip_label <- as_labeller(c(`gtot`='PSC', `vm`='Vm'))
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
    scale_fill_manual(values=color_map) + labs(tag='C')

cch <- arrow::read_feather(file.path(data_path, 'cch.feather'))
p4 <- cch %>%
    ggplot(aes(x=lags, color=resolution)) + geom_line(aes(y=cch)) +
    # facet_grid did not work with free_y here
    facet_wrap(vars(fct_rev(as.factor(input_rate))), scales="free_y", nrow=1) +
    coord_cartesian(xlim=c(-2.5, 2.5)) +
    scale_color_manual(values=color_map) + labs(tag='D')

fig <- ((p1 / (p2[[1]] | p2[[2]] | p2[[3]] | p2[[4]] | p2[[5]])) / p3) / p4
ggsave(file.path(data_path, 'fig.png'), fig)
