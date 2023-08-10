library(ggplot2)
library(arrow)
library(plotly)
library(jsonlite)
# TODO library(stringr)

library(argparser)
include('plots/parse_inputs.R')

wd = getwd()
data_path <- Sys.glob(file.path(wd, argv$source))

metadata <- fromJSON(file.path(dir, "metadata.json"))

tsim <- as.double(str_sub(metadata$duration, 1, -2))

#df_raster <- read.csv(data_path) %>%
df_raster <- read.csv('output_spikes.csv')#%>%
             #filter(time_ms>250 & time_ms<500)

rates <- df_raster %>%
  group_by(id) %>%
  summarise(rate=n()/tsim)

raster <- ggplot(df_raster, aes(x=time_ms, y=id)) +
          geom_point(shape=20, size=1) + theme_bw()

options("browser"="brave")
ggplotly(raster)
