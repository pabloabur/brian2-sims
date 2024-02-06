library(ggplot2)
library(jsonlite)
library(purrr)
library(dplyr)
library(wesanderson)
library(arrow)
library(patchwork)
library(stringr)

library(argparser)
include('plots/parse_inputs.R')

wd = getwd()
dir_list <- Sys.glob(file.path(wd, argv$source, '*/'))
data <- map(file.path(dir_list, 'metadata.json'), fromJSON)
acc <- map_dbl(data, ~.$accuracy)
size <- map_int(data, ~.$size)
prec <- map_chr(data, ~.$precision)
tbl <- tibble(accuracy=acc, size=size, precision=prec)

fig_lsm <- tbl %>%
    group_by(size, precision) %>%
    summarise(num=n(), mean_acc=mean(accuracy), sd_acc=sd(accuracy)) %>%
    mutate(se=sd_acc/sqrt(num)) %>%
    mutate(ic=se*qt(p=.05/2, df=num-1, lower.tail=F)) %>%
    ggplot(aes(x=size, y=mean_acc, color=precision, fill=precision)) +
    geom_line(aes(color=precision)) +
    geom_errorbar(aes(ymin=mean_acc-ic, ymax=mean_acc+ic), width=0) +
    #geom_ribbon(aes(ymin=mean_acc-ic, ymax=mean_acc+ic),
    #            alpha=0.3, linetype=0) +
    scale_color_manual(values=wes_palette('Moonrise1')[c(2, 4)]) +
    scale_fill_manual(values=wes_palette('Moonrise1')[c(2, 4)]) +
    ylim(.25, .75) + theme_bw() + theme(text=element_text(size=16)) +
    labs(y='LSM mean accuracy', fill='Bit-precision')

# Randomly chooses folder with data from linear SVM and ELM
dir_list <- sample(dir_list, 10)
svm <- map_dfr(file.path(dir_list, 'linear_acc.feather'), read_feather)
elm <- map_dfr(file.path(dir_list, 'elm_acc.feather'), read_feather)
fig_svm <- svm %>%
    group_by(regularization) %>%
    summarise(acc=mean(score)) %>%
    ggplot(aes(x=regularization, y=acc)) +
    geom_line() + scale_x_log10() + theme_bw() +
    theme(text=element_text(size=16)) + labs(y='SVM mean accuracy')
fig_elm <- elm %>%
    group_by(size) %>%
    summarise(acc=mean(score)) %>%
    ggplot(aes(x=size, y=acc)) +
    geom_line() + theme_bw() +
    theme(text=element_text(size=16)) + labs(y='ELM mean accuracy')

fig <- (fig_lsm / (fig_svm | fig_elm)) + plot_annotation(tag_levels='A')
ggsave(argv$dest, fig)
