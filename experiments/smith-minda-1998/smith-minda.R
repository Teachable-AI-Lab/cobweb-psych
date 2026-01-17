library(readr)
library(ggplot2)
library(tidyr)
library(dplyr)
library(cowplot)

exp_smith_minda <- read_csv("/Users/cmaclellan3/Projects/cobweb-psych/experiments/smith-minda-1998/exp_smith-minda_blocks10_nseeds5_epoch5.csv")
exp_smith_minda$stimulus <- as.factor(exp_smith_minda$stimulus)
exp_smith_minda$block <- as.factor(exp_smith_minda$block)
exp_smith_minda$seed <- as.factor(exp_smith_minda$seed)
exp_smith_minda$epoch <- as.factor(exp_smith_minda$epoch)
exp_smith_minda$category <- as.factor(exp_smith_minda$category)

# set up a column distinguishing stimulus 7 & 14 and others.
exp_smith_minda$type <- ifelse(exp_smith_minda$stimulus %in% c(7, 14), "Atypical", "Typical")
exp_smith_minda$type <- as.factor(exp_smith_minda$type)
summary(exp_smith_minda)

# dataframes for different methods:
#exp_leaf <- select(exp_smith_minda, -basic_A, -basic_B, -best_A, -best_B)
#exp_basic <- select(exp_smith_minda, -leaf_A, -leaf_B, -best_A, -best_B)
#exp_best <- select(exp_smith_minda, -leaf_A, -leaf_B, -basic_A, -basic_B)
#summary(exp_leaf)


# Plot
ggplot(exp_smith_minda, aes(x = block, y = pred_A, fill = category, shape = type, group=stimulus)) +
  stat_summary(fun = mean, geom = "line") +
  stat_summary(fun.data = mean_cl_boot, geom = "errorbar", width=0.15, alpha=0.7) +
  stat_summary(fun = mean, geom = "point", size=3) +
  scale_shape_manual(values = c(
    "Typical"  = 21,  # circle (fillable)
    "Atypical" = 24   # triangle (fillable)
  )) +
  scale_fill_manual(values = c(
    "A" = "white",   # unfilled
    "B" = "black"    # filled
  )) +
  labs(
    x = "Segment",
    y = "Probability of Category A",
    title = "Cobweb Model"
  ) +
  theme_classic(base_size = 15) +
  guides(
    fill = guide_legend(
      override.aes = list(
        shape = 21,
        color = "black"
      )
    )
  ) +
  scale_y_continuous(
    limits = c(0, 1),
    breaks = seq(0, 1, by = 0.2)
  )

# ggsave("/Users/cmaclellan3/Projects/cobweb-psych/experiments/smith-minda-1998/cobweb-exemplar-prototype.png")
