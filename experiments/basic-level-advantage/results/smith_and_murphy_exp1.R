library(dplyr)
library(tidyr)

exp1_smith_murphy<- read_csv("/Users/cmaclellan3/Projects/cobweb-psych/experiments/basic-level-advantage/results/smith_and_murphy_exp1.csv")
exp1_smith_murphy$`participant seed` <- as.factor(exp1_smith_murphy$`participant seed`)
exp1_smith_murphy$condition <- as.factor(exp1_smith_murphy$condition)
exp1_smith_murphy$category_level <- as.factor(exp1_smith_murphy$category_level)
exp1_smith_murphy$probability <- as.numeric(exp1_smith_murphy$probability)

exp1_smith_murphy <- exp1_smith_murphy %>%
  mutate(
    category_level = factor(
      category_level,
      levels = c("Subordinate", "Basic", "Superordinate")
    ),
    condition = factor(
      condition,
      levels = c("subordinate-first", "basic-first", "superordinate-first")
    )
  )

exp1_smith_murphy %>%
  group_by(condition, category_level) %>%
  summarise(
    mean_probability = mean(probability, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  pivot_wider(
    names_from = category_level,
    values_from = mean_probability
  )
