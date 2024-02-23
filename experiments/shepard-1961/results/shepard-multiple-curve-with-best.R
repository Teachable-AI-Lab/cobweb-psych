library(readr)
library(ggplot2)
library(tidyr)
library(dplyr)
library(cowplot)

exp_shepard_multiple <- read_csv("[your directory]/cobweb-psych/experiments/shepard-1961/results/exp_shepard_multiple_blocks23_nseeds5_epoch5_shuffled.csv")
summary(exp_shepard_multiple)
# have the data for block 1, 3, 5, 7, ..., 23 only
blocks_to_display <- c(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23)
exp_shepard_multiple <- exp_shepard_multiple[exp_shepard_multiple$block %in% blocks_to_display, ]

df_shepard_observation <- read_csv("/Users/xinlian/Library/Mobile Documents/com~apple~CloudDocs/硕士 我在GaTech 我学习 我快乐。/23 Fall/Courses/CS 7651 Human & ML/Project/Experiments/scripts-final/human_results_smith.csv")

# Get the values of task_type (1, 2, ..., 6):
task_types <- unique(exp_shepard_multiple$task_type)

# Create a list of plots for different task_type:
plot_list <- list()

# Create list of correlations for different method:
correlations_leaf <- list()
correlations_basic <- list()
correlations_best <- list()

# Draw separate plots for each task type:
for (task_type in task_types) {
  # Filter the dataframe based on the task type:
  df_task_type <- exp_shepard_multiple[exp_shepard_multiple$task_type == task_type, ]
  
  # Obtain the corresponding observed accuracies for the first 10 blocks:
  col_name <- paste0('type', task_type)
  probs_observed <- c(df_shepard_observation[1:12, col_name])
  probs_observed <- lapply(probs_observed, function(x) x / 100)  # divide by 100
  
  # Create a new column to assign the observed accuracies to the dataframe:
  # df_task_type$accuracy_observed <- mapply(function(index, vector) vector[index], 
  #                                          df_task_type$block, probs_observed)
  df_task_type$accuracy_observed <- mapply(function(index, vector) vector[index], 
                                           (df_task_type$block + 1) / 2, probs_observed)
  
  # Create pivot longer dataframe:
  df_long <- df_task_type %>% pivot_longer(
    cols=c('accuracy_observed', 'accuracy_leaf', 'accuracy_basic', 'accuracy_best'),
    names_to='derivation',
    values_to='accuracy'
  )
  df_long <- df_long %>%
    mutate(derivation = case_when(
      derivation == "accuracy_observed" ~ "observed",
      derivation == "accuracy_leaf" ~ "leaf",
      derivation == "accuracy_basic" ~ "basic",
      derivation == "accuracy_best" ~ "best",
      TRUE ~ as.character(derivation)  # Keep other values unchanged
    ))
  
  # Drop the "best" accuracies:
  # df_long <- subset(df_long, derivation != "best")
  
  # Draw the plots
  p_shepard_task <- ggplot(df_long, aes(x = block, y = accuracy, color = derivation,
                                       linetype = derivation, fill = derivation, group = derivation)) +
    geom_errorbar(stat = "summary",
                  fun.data = "mean_cl_boot",
                  linetype = 'solid',
                  width = 0.2) +
    stat_summary(fun.data = "mean_cl_boot",
                 geom = "line",
                 linewidth = 2,
                 alpha = 0.5) +
    labs(x = '# of Iterations', y = 'Accuracy',
         title = sprintf("Task %d", task_type)) +
    # labs(title = sprintf("Task %d", task_type)) +
    
    theme_minimal() +
    theme(text = element_text(size=15)) +
    theme(axis.title.x = element_blank(), axis.title.y = element_blank()) +
    scale_color_manual(values = c('observed' = '#ca0020',
    'leaf' = '#fdae61', 'basic' = '#7fcdbb', 'best' = '#2c7fb8')) +
    # scale_color_manual(values = c('observed' = '#ca0020', 
    #                               'leaf' = '#fdae61', 'basic' = '#7fcdbb')) +
    scale_linetype_manual(values = c('observed' = 'solid',
    'leaf' = 'solid', 'basic' = 'solid', 'best' = 'solid')) +
    # scale_linetype_manual(values = c('observed' = 'solid', 
    #                                  'leaf' = 'solid', 'basic' = 'solid')) +
    theme(plot.title = element_text(hjust = 0.5),
          # legend.position = c(.9, .2),
          legend.position = "none",
          legend.key.size = unit(0.5, "lines"),
          plot.margin = unit(c(1, 0.5, 0.5, 0.5), "lines")) 
    # scale_x_discrete(breaks=as.character(blocks_to_display))
  
  plot_list[[task_type]] <- p_shepard_task
  
  # Compute correlation coefficient:
  df_summary<- ggplot_build(p_shepard_task)$data[[2]]
  probs_basic <- df_summary[df_summary['group'] == 1, 'y']
  probs_best <- df_summary[df_summary['group'] == 2, 'y']
  probs_leaf <- df_summary[df_summary['group'] == 3, 'y']
  probs_observed <- df_summary[df_summary['group'] == 4, 'y']
  correlation_leaf <- cor(probs_observed, probs_leaf)
  correlation_basic <- cor(probs_observed, probs_basic)
  correlation_best <- cor(probs_observed, probs_best)
  correlations_leaf[task_type] <- correlation_leaf
  correlations_basic[task_type] <- correlation_basic
  correlations_best[task_type] <- correlation_best
  
}

legend <- cowplot::get_legend(plot_list[[1]])
# Plotting multiple figures in a grid layout
multiplot <- cowplot::plot_grid(plotlist = plot_list, ncol = 3)
# multiplot <- grid + labs(x = "Common X-axis Label", y = "Common Y-axis Label")

# Integrating legends:
# legend <- cowplot::get_legend(plot_list[[1]])

# Add an overall title
overall_title <- ggdraw() +
  draw_label("Accuracy with # of Iterations, Shepard (1961)", size = 20, fontface = "bold")
final_plot <- plot_grid(
  # plot_grid(NULL, get_legend(plot_list[[1]]), nrow=1),
  # cowplot::get_legend(plot_list[[1]]),
  plot_grid(overall_title, multiplot, ncol = 1, rel_heights = c(0.1, 1)),
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

unlist(correlations_leaf)
# Unshuffled: NA 0.8613299 0.8441499        NA 0.9143803 0.8415694
# Shuffled: NA 0.9290234 0.7514303        NA 0.9842239 0.8835008
unlist(correlations_basic)
# Unshuffled: NA  0.9204164  0.8547835  0.8997698  0.9472460 -0.8415694
# Shuffled: 0.9808949  0.9317226  0.8409413  0.9103406  0.9840743 -0.3752031
unlist(correlations_best)
# Unshuffled: NA 0.9131917 0.8665476 0.8997698 0.9483947 0.8415694
# Shuffled: 0.9808949 0.9406132 0.8471873 0.8959408 0.9826809 0.9284294

