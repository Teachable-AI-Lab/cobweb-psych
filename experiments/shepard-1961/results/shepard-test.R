library(readr)
library(ggplot2)
library(tidyr)
library(dplyr)
library(cowplot)

exp_shepard_test <- read_csv("[your directory]/cobweb-psych/experiments/shepard-1961/results/exp_shepard_test_nseeds5_epoch5.csv")
summary(exp_shepard_test)

df_shepard_observation <- read_csv("[your dictrectory]/cobweb-psych/experiments//shepard-1961/human_results_smith.csv")
df_shepard_observation$block <- as.character(df_shepard_observation$block)  # all become strings

# The name of the blocks:
blocks <- unique(df_shepard_observation$block)

# Create a list of plots for different block:
plot_list <- list()

# Create list of correlations for different method:
correlations_leaf <- list()
correlations_basic <- list()
correlations_best <- list()

count <- 1
# Draw separate plots for each testing block:
for (block in blocks) {
  df_shepard_block <- exp_shepard_test
  # Obtain the corresponding observed accuracies for each block:
  # col_name <- paste0('type', task_type)
  probs_observed <- df_shepard_observation[df_shepard_observation$block == block, ]
  probs_observed <- c(probs_observed[-c(1, length(probs_observed))])
  probs_observed <- lapply(probs_observed, function(x) x / 100)  # divide by 100
  probs_observed <- unlist(probs_observed, use.names = FALSE)
  
  # Create a new column to assign the observed accuracies to the dataframe
  # df_shepard_block$accuracy_observed <- mapply(function(index, vector) vector[index], 
                                               # df_shepard_block$task_type, probs_observed)
  df_shepard_block <- df_shepard_block %>%
    mutate(accuracy_observed = probs_observed[task_type])
  
  # Create pivot longer dataframe:
  df_long <- df_shepard_block %>% pivot_longer(
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
  
  # Draw the plots
  p_shepard_block <- ggplot(df_long, aes(x = task_type, y = accuracy, color = derivation,
                                        linetype = derivation, fill = derivation, group = derivation)) +
    geom_errorbar(stat = "summary",
                  fun.data = "mean_cl_boot",
                  linetype = 'solid',
                  width = 0.2) +
    stat_summary(fun.data = "mean_cl_boot",
                 geom = "line",
                 size = 2,
                 alpha = 0.5) +
    labs(x = 'Task Type', y = 'Accuracy', 
         title = paste("Block ", block)) +
    
    theme_minimal() +
    theme(text = element_text(size=15)) +
    scale_color_manual(values = c('observed' = '#ca0020', 
                                  'leaf' = '#fdae61', 'basic' = '#7fcdbb', 'best' = '#2c7fb8')) +
    scale_linetype_manual(values = c('observed' = 'solid', 
                                     'leaf' = 'solid', 'basic' = 'solid', 'best' = 'solid')) +
    theme(plot.title = element_text(hjust = 0.5),
          # legend.position = c(.9, .2),
          legend.position = "none",
          legend.key.size = unit(0.5, "lines"),
          plot.margin = unit(c(1, 0.5, 0.5, 0.5), "lines"))
  
  plot_list[[count]] <- p_shepard_block
  
  # Compute correlation coefficient:
  df_summary<- ggplot_build(p_shepard_block)$data[[2]]
  probs_basic <- df_summary[df_summary['group'] == 1, 'y']
  probs_best <- df_summary[df_summary['group'] == 2, 'y']
  probs_leaf <- df_summary[df_summary['group'] == 3, 'y']
  probs_observed <- df_summary[df_summary['group'] == 4, 'y']
  correlation_leaf <- cor(probs_observed, probs_leaf)
  correlation_basic <- cor(probs_observed, probs_basic)
  correlation_best <- cor(probs_observed, probs_best)
  correlations_leaf[count] <- correlation_leaf
  correlations_basic[count] <- correlation_basic
  correlations_best[count] <- correlation_best
  
  count <- count + 1
}


# Plotting multiple figures in a grid layout
multiplot <- cowplot::plot_grid(plotlist = plot_list, ncol = 5)

# Integrating legends:
# legend <- cowplot::get_legend(plot_list[[1]])

# Add an overall title
overall_title <- ggdraw() +
  draw_label("Accuracy on Each Task Type, Shepard (1961)", size = 20, fontface = "bold")
final_plot <- plot_grid(
  # plot_grid(NULL, get_legend(plot_list[[1]]), nrow=1),
  # cowplot::get_legend(plot_list[[1]]),
  plot_grid(overall_title, multiplot, ncol = 1, rel_heights = c(0.1, 1)),
  ncol = 1,
  rel_heights = c(0.1, 1.5)
)

final_plot

unlist(correlations_leaf)
# 0.5731811 0.6423158 0.6325522 0.5112455 0.5754435 0.6482860 0.5985325 0.7489880 0.4645106
# 0.7538609 0.7092915 0.7812033 0.6638908
unlist(correlations_basic)
# 0.8045708 0.7761783 0.7736219 0.6748184 0.7347818 0.8427705 0.7945968 0.8852508 0.7082851
# 0.9034724 0.8428033 0.9407556 0.8318373
unlist(correlations_best)
# 0.8000966 0.7603006 0.7631733 0.6610901 0.7299831 0.8408034 0.7990080 0.8843993 0.7133972
# 0.9028569 0.8438931 0.9435894 0.8271186
