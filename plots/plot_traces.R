library(arrow)
library(dplyr)
library(ggplot2)
library(plotly)
library(htmlwidgets)

df_traces <- arrow::read_feather('output_traces.feather')
df_input <- arrow::read_feather('input_spikes.feather')
df_output <- arrow::read_feather('output_spikes.feather')
df_rec <- arrow::read_feather('rec_spikes.feather')

p <- ggplot(df_traces, aes(x=time_ms)) +
  geom_line(aes(y=Vm_mV, color=id)) +
  geom_line(aes(y=Vthr_mV, color=id))
p <- ggplotly(p)

fig1 <- plot_ly(x=df_input$time_ms, y=df_input$id, type = 'scatter', mode='markers', name='input')
fig2 <- plot_ly(x=df_rec$time_ms, y=df_rec$id, type = 'scatter', mode='markers', name='recurrent')
fig3 <- plot_ly(x=df_output$time_ms, y=df_output$id, type = 'scatter', mode='markers', name='output')
fig <- subplot(p, fig1, fig2, fig3, nrows = 4, shareX = T)
fig
#saveWidget(fig, 'spikes.html')

# If working with membrane traces, for example
pc64 <- arrow::read_feather('test_memb.feather')
pc8 <- arrow::read_feather('test_memb.feather')
joined <- inner_join(pc8, pc64, by="time_ms")
library(tidyr)
joined_lng <- pivot_longer(joined, 2:3, names_to = "prec", values_to = "Vm")
p <- ggplot(joined_lng, aes(x=time_ms, y=Vm, color=prec)) +
  geom_line()
