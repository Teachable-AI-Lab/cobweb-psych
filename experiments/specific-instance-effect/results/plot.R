library(readr)
library(ggplot2)
library(dplyr)

hayes_roth_orig<- read_csv("/Users/cmaclellan3/Projects/cobweb-psych/experiments/specific-instance-effect/results/classification_ratings.csv")
hayes_roth_orig$code <- as.factor(hayes_roth_orig$code)
hayes_roth_data<- read_csv("/Users/cmaclellan3/Projects/cobweb-psych/experiments/specific-instance-effect/results/exp_specific_instance_discrete.csv")
hayes_roth_data$code <- as.factor(hayes_roth_data$code)

hayes_roth_data <- hayes_roth_data %>%
  left_join(hayes_roth_orig, by = "code")

ggplot(hayes_roth_data, aes(x = code, y = p_club1)) +
  stat_summary(fun = mean, geom = "bar") +
  stat_summary(fun.data = mean_cl_boot, geom = "errorbar", width=0.15, alpha=0.7)

hayes_roth_data <- hayes_roth_data %>%
  mutate(club_tag = interaction(club,tag,freq, sep = "-"))

hayes_roth_data$freq <- as.factor(hayes_roth_data$freq)
ggplot(hayes_roth_data[hayes_roth_data$tag != "Other",], aes(x = reorder(code, classification_rating), y = p_club1, shape=as.factor(freq))) +
  stat_summary(fun = mean, geom = "point") +
  stat_summary(fun.data = mean_cl_boot, geom = "errorbar", width=0.15, alpha=0.7) +
  theme_classic()

avg_rating_by_club <- hayes_roth_data %>%
  group_by(club_tag) %>%
  summarise(
    avg_classification_rating = mean(classification_rating, na.rm = TRUE),
    .groups = "drop"
  )

hayes_roth_data <- hayes_roth_data %>%
  mutate(club_freq = interaction(club, freq, sep = "-"))

ggplot(hayes_roth_data[hayes_roth_data$tag != "Other",], aes(x = reorder(club_tag, -classification_rating), y = p_club1, fill=club_freq, shape=club_freq)) +
  stat_summary(fun.data = mean_cl_boot, geom = "errorbar", width=0.15, alpha=0.7) +
  stat_summary(fun = mean, geom = "point", size=3) +
    theme_classic() +
  labs(
    x = "Types of Items Ordered by Avg. Human Classification Ratings for `Club 1`",
    y = "Model Probability of `Club 1` Category",
    shape = "Club × Frequency",
    fill = "Club × Frequency"
  ) +
  scale_shape_manual(
    name = "Club × Frequency",
    values = c(
    "1-1"  = 21,  # circle (fillable)
    "1-0" = 24,   # triangle (fillable)
    "1-10"  = 22,  # square (fillable)
    "2-1"  = 21,  # circle (fillable)
    "2-0" = 24,   # triangle (fillable)
    "2-10"  = 22  # square (fillable)
  ),
  labels = c(
    "1-0"  = "Club 1 - Never seen",
    "1-1"  = "Club 1 - Seen once",
    "1-10" = "Club 1 - Seen 10×",
    "2-0"  = "Club 2 - Never seen",
    "2-1"  = "Club 2 - Seen once",
    "2-10" = "Club 2 - Seen 10×"
    
  )) +
  scale_fill_manual(
    name = "Club × Frequency",
    values = c(
    "1-1" = "white",   # unfilled
    "2-1" = "black",    # filled
    "1-0" = "white",   # unfilled
    "2-0" = "black",    # filled
    "1-10" = "white",   # unfilled
    "2-10" = "black"    # filled
    ),
    labels = c(
      "1-0"  = "Club 1 - Never seen",
      "1-1"  = "Club 1 - Seen once",
      "1-10" = "Club 1 - Seen 10×",
      "2-0"  = "Club 2 - Never seen",
      "2-1"  = "Club 2 - Seen once",
      "2-10" = "Club 2 - Seen 10×"
    )) +
  guides(
    shape    = guide_legend(order = 2, ncol = 2, byrow = TRUE),
    fill     = guide_legend(order = 2, ncol = 2, byrow = TRUE)
  ) +
  theme(
    legend.position = c(0.5, 0.65),
    legend.justification = c("right", "bottom"),
    legend.background = element_rect(fill = "white", color = "black")
  ) +
  scale_x_discrete(labels = c(
    "2-Prototype-0" = "Prototype\n(Freq=0)",
    "2-1Transform-10" = "1 Transform\n(Freq=10)",
    "2-1Transform-1" = "1 Transform\n(Freq=1)",
    "2-2Transform-1" = "2 Transforms\n(Freq=1)",
    "1-Prototype-0" = "Prototype\n(Freq=0)",
    "1-1Transform-10" = "1 Transform\n(Freq=10)",
    "1-1Transform-1" = "1 Transform\n(Freq=1)",
    "1-2Transform-1" = "2 Transforms\n(Freq=1)"
  ))

#ggplot(hayes_roth_data[hayes_roth_data$tag != "Other",], aes(x = classification_rating, y = p_club1)) +
#  stat_summary(fun = mean, geom = "point") +
#  stat_summary(fun.data = mean_cl_boot, geom = "errorbar", width=0.15, alpha=0.7)
