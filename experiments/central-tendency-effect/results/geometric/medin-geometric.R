library(dplyr)
library(readr)
library(ggplot2)
library(tidyr)

exp_medin_geo <- read_csv("[your directory]/cobweb-psych/experiments/medin-schaffer-1978/results/geometric/exp_medin_type-geometric_nseeds40_epoch5_face_prediction.csv")
# The predictions for both types of stimuli should be the same in the presentation. 
# We here adopt the predictions from the face stimuli because they have better correlations.
summary(exp_medin_geo)


"
Plot the probabilities: 
x-axis is the nbr of each stimulus (16 in total),
y-axis is the prob.
Draw in three categories - leaf, basic, best, along with observed prob.
"

df_long <- exp_medin_geo %>% pivot_longer(
  cols=c('observed_probs', 'predicted_probs_leaf', 'predicted_probs_basic', 'predicted_probs_best'),
  names_to='derivation',
  values_to='probs'
)

df_long <- df_long %>%
  mutate(derivation = case_when(
    derivation == "observed_probs" ~ "observed",
    derivation == "predicted_probs_leaf" ~ "leaf",
    derivation == "predicted_probs_basic" ~ "basic",
    derivation == "predicted_probs_best" ~ "best",
    TRUE ~ as.character(derivation)  # Keep other values unchanged
  ))

summary(df_long)

p_medin_geo <- ggplot(df_long, aes(x = stimulus, y = probs, color = derivation,
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
       title = 'Observed and Predicted Probabilities for Medin and Schaffer (1978),\nGeometric Stimuli') +
  
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
print(p_medin_geo)

# Output the stat summary shown in the plots
df_summary_geo<- ggplot_build(p_medin_geo)$data[[2]]

# Calculate the correlation coefficient:
probs_basic <- df_summary_geo[df_summary_geo['group'] == 1, 'y']
probs_best <- df_summary_geo[df_summary_geo['group'] == 2, 'y']
probs_leaf <- df_summary_geo[df_summary_geo['group'] == 3, 'y']
probs_observed <- df_summary_geo[df_summary_geo['group'] == 4, 'y']
correlation_leaf <- cor(probs_observed, probs_leaf)  # 0.7680 | 0.7800 with face prediction
correlation_basic <- cor(probs_observed, probs_basic)  # 0.7129 | 0.7238 with face prediction
correlation_best <- cor(probs_observed, probs_best)  # 0.6685 | 0.6791 with face prediction

# Compare the predicted probabilities for Stimulus 4 and 7:
df_summary_geo_4 <- df_summary_geo[df_summary_geo['x'] == 4, ]
c(df_summary_geo_4[,'y'])
# 0.8342798 0.8216101 0.7425000 0.7800000
df_summary_geo_7 <- df_summary_geo[df_summary_geo['x'] == 7, ]
c(df_summary_geo_7[,'y'])
# 0.8214048 0.7983333 0.7500000 0.8800000
