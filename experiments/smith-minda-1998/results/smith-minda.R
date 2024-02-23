library(readr)
library(ggplot2)
library(tidyr)
library(dplyr)
library(cowplot)

exp_smith_minda <- read_csv("[your directory]/cobweb-psych/experiments/smith-minda-1998/exp_smith-minda_blocks10_nseeds5_epoch5.csv")
exp_smith_minda$stimulus <- as.factor(exp_smith_minda$stimulus)
exp_smith_minda$block <- as.factor(exp_smith_minda$block)
exp_smith_minda$seed <- as.factor(exp_smith_minda$seed)
exp_smith_minda$epoch <- as.factor(exp_smith_minda$epoch)
exp_smith_minda$category <- as.factor(exp_smith_minda$category)

# set up a column distinguishing stimulus 7 & 14 and others.
exp_smith_minda$normal <- ifelse(exp_smith_minda$stimulus %in% c(7, 14), "N", "Y")
exp_smith_minda$normal <- as.factor(exp_smith_minda$normal)
summary(exp_smith_minda)

# dataframes for different methods:
exp_leaf <- select(exp_smith_minda, -basic_A, -basic_B, -best_A, -best_B)
exp_basic <- select(exp_smith_minda, -leaf_A, -leaf_B, -best_A, -best_B)
exp_best <- select(exp_smith_minda, -leaf_A, -leaf_B, -basic_A, -basic_B)
summary(exp_leaf)


# Plot 1: Leaf
p_leaf <- ggplot(exp_leaf, aes(x = block, y = leaf_A, color = category, linetype = normal, 
                               fill = stimulus, group = stimulus)) +
  geom_errorbar(stat = "summary",
                fun.data = "mean_cl_boot",
                linetype = 'solid',
                width = 0.2) +
  stat_summary(fun.data = "mean_cl_boot",
               geom = "line",
               linewidth = 2,
               alpha = 0.5) +
  # facet_wrap(~stimulus) +
  labs(x = '# of Blocks', 
       y = 'Predicted Probability on Category A or B',
       title = "Leaf Prediction") +
  # labs(title = sprintf("Task %d", task_type)) +
  
  theme_minimal() +
  theme(text = element_text(size=15)) +
  theme(axis.title.x = element_blank(), axis.title.y = element_blank()) +
  scale_color_manual(values = c('A' = 'red', 'B' = 'blue')) +
  scale_linetype_manual(values = c('N' = 'dashed', 'Y' = 'solid')) +
  theme(plot.title = element_text(hjust = 0.5),
        # legend.position = c(.9, .2),
        legend.position = "none",
        legend.key.size = unit(0.5, "lines"),
        plot.margin = unit(c(1, 0.5, 0.5, 0.5), "lines")) 

p_leaf
# scale_x_discrete(breaks=as.character(blocks_to_display))

# Figure 2: basic level
p_basic <- ggplot(exp_basic, aes(x = block, y = basic_A, color = category, linetype = normal, 
                               fill = stimulus, group = stimulus)) +
  geom_errorbar(stat = "summary",
                fun.data = "mean_cl_boot",
                linetype = 'solid',
                width = 0.2) +
  stat_summary(fun.data = "mean_cl_boot",
               geom = "line",
               linewidth = 2,
               alpha = 0.5) +
  # facet_wrap(~stimulus) +
  labs(x = '# of Blocks', 
       y = 'Predicted Probability on Category A or B',
       title = "Basic Prediction") +
  # labs(title = sprintf("Task %d", task_type)) +
  
  theme_minimal() +
  theme(text = element_text(size=15)) +
  theme(axis.title.x = element_blank(), axis.title.y = element_blank()) +
  scale_color_manual(values = c('A' = 'red', 'B' = 'blue')) +
  scale_linetype_manual(values = c('N' = 'dashed', 'Y' = 'solid')) +
  theme(plot.title = element_text(hjust = 0.5),
        # legend.position = c(.9, .2),
        legend.position = "none",
        legend.key.size = unit(0.5, "lines"),
        plot.margin = unit(c(1, 0.5, 0.5, 0.5), "lines")) 

p_basic

# Figure 3: best level
p_best <- ggplot(exp_best, aes(x = block, y = best_A, color = category, linetype = normal, 
                                 fill = stimulus, group = stimulus)) +
  geom_errorbar(stat = "summary",
                fun.data = "mean_cl_boot",
                linetype = 'solid',
                width = 0.2) +
  stat_summary(fun.data = "mean_cl_boot",
               geom = "line",
               linewidth = 2,
               alpha = 0.5) +
  # facet_wrap(~stimulus) +
  labs(x = '# of Blocks', 
       y = 'Predicted Probability on Category A or B',
       title = "Best Prediction") +
  # labs(title = sprintf("Task %d", task_type)) +
  
  theme_minimal() +
  theme(text = element_text(size=15)) +
  theme(axis.title.x = element_blank(), axis.title.y = element_blank()) +
  scale_color_manual(values = c('A' = 'red', 'B' = 'blue')) +
  scale_linetype_manual(values = c('N' = 'dashed', 'Y' = 'solid')) +
  theme(plot.title = element_text(hjust = 0.5),
        # legend.position = c(.9, .2),
        legend.position = "none",
        legend.key.size = unit(0.5, "lines"),
        plot.margin = unit(c(1, 0.5, 0.5, 0.5), "lines")) 

p_best




# make the aggregated plots
plot_list <- list()
plot_list[[1]] <- p_leaf
plot_list[[2]] <- p_basic
plot_list[[3]] <- p_best

legend <- cowplot::get_legend(plot_list[[1]])
# Plotting multiple figures in a grid layout
multiplot <- cowplot::plot_grid(plotlist = plot_list, ncol = 3)
# multiplot <- grid + labs(x = "Common X-axis Label", y = "Common Y-axis Label")

# Integrating legends:
# legend <- cowplot::get_legend(plot_list[[1]])

# Add an overall title
# overall_title <- ggdraw() +
  # draw_label("Predicted Probability on Category A", size = 20, fontface = "bold")
final_plot <- plot_grid(
  # plot_grid(NULL, get_legend(plot_list[[1]]), nrow=1),
  # cowplot::get_legend(plot_list[[1]]),
  # plot_grid(overall_title, multiplot, ncol = 1, rel_heights = c(0.1, 1)),
  plot_grid(multiplot, ncol = 1, rel_heights = c(0.1, 1)),
  ncol = 1,
  rel_heights = c(0.1, 1.5)
  )

# Add one final legend
# legend <- get_legend(plot_list[[1]])
# grid_with_legend <- final_plot + theme(legend.position = "none") +
#   theme(plot.margin = unit(c(0, 0, -1.5, 0), "lines"))  # Adjust the bottom margin
# # Print the grid with the combined legend
# print(grid_with_legend + draw_plot(legend, x = 0.5, y = -0.2))

final_plot
# legend <- cowplot::get_legend(plot_list[[1]])
# legend


