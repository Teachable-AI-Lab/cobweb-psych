library(dplyr)
library(readr)
library(ggplot2)
library(tidyr)

exp_medin_face <- read_csv("[your directory]/cobweb-psych/experiments/medin-schaffer-1978/results/face/exp_medin_type-face_nseeds40_epoch5_geometric_predicion.csv")
summary(exp_medin_face)


"
Plot the probabilities: 
x-axis is the nbr of each stimulus (16 in total),
y-axis is the prob.
Draw in three categories - leaf, basic, best, along with observed prob.
"

# First pivot the dataframe:
# create a new column 'probs', 
# its values are assigned by the ones in columns 'observed_probs', 'predicted_probs_leaf', ...
# along with a new category column 'derivation' whose values are assigned by these column names
df_long <- exp_medin_face %>% pivot_longer(
  cols=c('observed_probs', 'predicted_probs_leaf', 'predicted_probs_basic', 'predicted_probs_best'),
  names_to='derivation',
  values_to='probs'
)

# change the category values of the column 'derivation'
df_long <- df_long %>%
  mutate(derivation = case_when(
    derivation == "observed_probs" ~ "observed",
    derivation == "predicted_probs_leaf" ~ "leaf",
    derivation == "predicted_probs_basic" ~ "basic",
    derivation == "predicted_probs_best" ~ "best",
    TRUE ~ as.character(derivation)  # Keep other values unchanged
  ))

summary(df_long)

# plot:
p_medin_face <- ggplot(df_long, aes(x = stimulus, y = probs, color = derivation,
                                   linetype = derivation, fill = derivation, group = derivation)) +
  geom_errorbar(stat = "summary",
                fun.data = "mean_cl_boot",
                linetype = 'solid',
                width = 0.2) +
  stat_summary(fun.data = "mean_cl_boot",
               geom = "line",
               size = 2,
               alpha = 0.5) +
  labs(x = 'Stimulus No.', y = 'Probability', 
       title = 'Observed and Predicted Probabilities for Medin and Schaffer (1978),\nFace Stimuli') +
  
  theme_minimal() +
  theme(text = element_text(size=22)) +
  scale_color_manual(values = c('observed' = '#ca0020', 
                                'leaf' = '#fdae61', 'basic' = '#7fcdbb', 'best' = '#2c7fb8')) +
  scale_linetype_manual(values = c('observed' = 'solid', 
                                   'leaf' = 'solid', 'basic' = 'solid', 'best' = 'solid')) +
  theme(plot.title = element_text(hjust = 0.5),
        legend.position = c(.9, .2),
        legend.key.size = unit(1.5, "lines"),
        plot.margin = unit(c(1, 0.5, 0.5, 0.5), "lines"))  
# scale_y_continuous(
#   breaks = c(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0))
print(p_medin_face)

# Output the stat summary shown in the plots
df_summary_face<- ggplot_build(p_medin_face)$data[[2]]

# Calculate the correlation coefficient:
probs_basic <- df_summary_face[df_summary_face['group'] == 1, 'y']
probs_best <- df_summary_face[df_summary_face['group'] == 2, 'y']
probs_leaf <- df_summary_face[df_summary_face['group'] == 3, 'y']
probs_observed <- df_summary_face[df_summary_face['group'] == 4, 'y']
correlation_leaf <- cor(probs_observed, probs_leaf)  # 0.8004
correlation_basic <- cor(probs_observed, probs_basic)  # 0.8241
correlation_best <- cor(probs_observed, probs_best)  # 0.8153

# Compare the predicted probabilities for Stimulus 4 and 7:
df_summary_face_4 <- df_summary_face[df_summary_face['x'] == 4, ]
c(df_summary_face_4[,'y'])
# 0.8374464 0.8235804 0.7350000 0.9700000
df_summary_face_7 <- df_summary_face[df_summary_face['x'] == 7, ]
c(df_summary_face_7[,'y'])
# 0.8256964 0.8018750 0.7500000 0.9700000

