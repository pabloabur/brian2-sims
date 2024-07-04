suppressPackageStartupMessages(library(xtable))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(stringr))
suppressPackageStartupMessages(library(wesanderson))
suppressPackageStartupMessages(library(latex2exp))
suppressPackageStartupMessages(library(patchwork))
suppressPackageStartupMessages(library(purrr))
suppressPackageStartupMessages(library(stringr))
suppressPackageStartupMessages(library(jsonlite))

library(argparser)
include('plots/parse_inputs.R')

color_map <- wes_palette('Zissou1')

wd = getwd()
data_path <- Sys.glob(file.path(wd, argv$source, "*"))
data_path <- data_path[str_detect(data_path, ".png", negate=T)]
op_type <- if (length(data_path) == 3) "addition" else "multiplication"
metadata <- map(file.path(data_path, "metadata.json"), fromJSON)
selected_folders  <- map_chr(metadata, \(x) x$sim_type)

sim_id <- match("standard", selected_folders)

df_ops <- read.csv(file.path(data_path[sim_id], "probs.csv"))
df_ops <- df_ops %>%
    # operations with zeros do not affect results, so it is removed
    filter(lambda != 8)

if (op_type == "addition") {
    table_1 <- df_ops %>%
        filter(val_x==0.015625) %>%
        select(-c(lambda, likely_count, unlikely_count)) %>%
        slice_sample(n=10) %>%
        mutate(L = if_else(L <= 0, NA, L))

    print(xtable(table_1, digits=c(0, 9, 9, 4, 4, 2, 3, 9, 9)))

    table_2 <- df_ops %>%
        filter(val_x==0.001953125) %>%
        select(-c(likely_count, unlikely_count)) %>%
        slice_sample(n=10) %>%
        mutate(L = if_else(L <= 0, NA, L))

    print(xtable(table_2, digits=c(0, 9, 9, 4, 4, 2, 2, 3, 9, 9)))
} else {
    table_1 <- df_ops %>%
        select(-c(lambda, delta, L, likely_count, unlikely_count)) %>%
        slice_sample(n=10)

    print(xtable(table_1, digits=c(0, 9, 9, 4, 4, 9, 9)))
}

#print(xtable(df_ops, digits=c(0, 9, 9, 4, 4, 2, 2, 3, 9, 9, 0, 0)), type="html", file=str_replace(argv$dest, "fig.png", "table.html"))

if (op_type == "addition"){
    df_L <- df_ops %>%
        select(lambda, delta, L) %>%
        filter(delta+lambda>7) %>%
        mutate(L = if_else(L <= 0, 0, L)) %>%
        distinct()
    L_heatmap <- ggplot(df_L, aes(delta, lambda, fill=L)) +
        scale_fill_gradientn(colors=color_map, name=TeX(r'($L$)')) +
        labs(x=TeX(r'($\delta$)'), y=TeX(r'($\lambda$)')) + theme_bw() + geom_tile() +
        theme(legend.position="top", text = element_text(size=16),
              legend.text=element_text(angle=45))
}

my_breaks<-c(1, 10, 100, 1000)
df_scatter <- df_ops %>%
    # This removes a negative case that is never actually observed
    filter(!(p_round_closer_ideal == 1 & p_round_closer_sim == 0))
probs_scatter <- ggplot(df_scatter, aes(x=p_round_closer_ideal, y=p_round_closer_sim)) +
    scale_fill_gradientn(name = "count", trans = "log", colors=color_map,
                        breaks = my_breaks, labels = my_breaks) +
    geom_smooth(colour='black', method=lm) + geom_bin2d(bins=25) +
    geom_abline(slope=1, intercept=0, linetype="dashed", color="black") +
    theme_bw() + labs(x=TeX(r'($P_i$)'), y=TeX(r'($P_s$)')) +
    theme(legend.position="top", text = element_text(size=16),
          legend.text=element_text(angle=45))

if (op_type == "addition"){
    sim_id <- match("stickier", selected_folders)
    df_sticky <- read.csv(file.path(data_path[sim_id], "probs.csv"))
    df_sticky <- df_sticky %>%
        filter(lambda != 8) %>%
        filter(!(p_round_closer_ideal == 1 & p_round_closer_sim == 0))
    probs_sticky <- ggplot(df_sticky, aes(x=p_round_closer_ideal, y=p_round_closer_sim)) +
        scale_fill_gradientn(name = "count", trans = "log", colors=color_map,
                            breaks = my_breaks, labels = my_breaks) +
        geom_smooth(colour='black', method=lm) + geom_bin2d(bins=25) +
        geom_abline(slope=1, intercept=0, linetype="dashed", color="black") +
        theme_bw() + labs(x=TeX(r'($P_i$)'), y=TeX(r'($P_s$)')) +
        theme(legend.position="none", text = element_text(size=16))

    sim_id <- match("cutoff", selected_folders)
    df_cut <- read.csv(file.path(data_path[sim_id], "probs.csv"))
    df_cut <- df_cut %>%
        filter(lambda != 8) %>%
        filter(!(p_round_closer_ideal == 1 & p_round_closer_sim == 0))
    probs_cut <- ggplot(df_cut, aes(x=p_round_closer_ideal, y=p_round_closer_sim)) +
        scale_fill_gradientn(name = "count", trans = "log", colors=color_map,
                            breaks = my_breaks, labels = my_breaks) +
        geom_smooth(colour='black', method=lm) + geom_bin2d(bins=25) +
        geom_abline(slope=1, intercept=0, linetype="dashed", color="black") +
        theme_bw() + labs(x=TeX(r'($P_i$)'), y=TeX(r'($P_s$)')) +
        theme(legend.position="none", text = element_text(size=16))
}

if (op_type == "addition"){
    fig_probs <- L_heatmap + probs_scatter + probs_sticky + probs_cut +
        plot_annotation(tag_levels='A')
} else {
    fig_probs <- probs_scatter
}

ggsave(argv$dest, fig_probs)
