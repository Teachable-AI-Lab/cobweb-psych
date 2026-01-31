library(ggrepel)
library(readr)
library(ggplot2)
library(tidyr)
library(dplyr)
library(cowplot)
library(tibble)

#########
# Experiment 1 Medin and Schaffer 1978
#########

exp1_medin <- read_csv("/Users/cmaclellan3/Projects/cobweb-psych/experiments/central-tendency-effect/results/medin_exp1_anderson_pred.csv")
exp1_errors <- read_csv("/Users/cmaclellan3/Projects/cobweb-psych/experiments/central-tendency-effect/results/exp_medin_exp1_errors.csv")
exp1_pred <- read_csv("/Users/cmaclellan3/Projects/cobweb-psych/experiments/central-tendency-effect/results/exp_medin_exp1_results.csv")

# Reorder Human ratings to relative to class A
exp1_medin[exp1_medin$Item == "6",]$Rating <- 4.8001
exp1_medin$Rating_adj <- exp1_medin$Rating
exp1_medin[exp1_medin$Category == "B",]$Rating_adj <- -1 * (exp1_medin[exp1_medin$Category == "B",]$Rating - 3.5) + 3.5

# Preprocess
exp1_medin <- exp1_medin %>%
  select(Item, Category, Type, Prediction, Rating_adj)
exp1_medin$Model <- "RMC"
exp1_medin$Subject <- "Avg"
exp1_medin$Item <- as.factor(exp1_medin$Item)
exp1_medin$Category <- as.factor(exp1_medin$Category)
exp1_medin$Type <- as.factor(exp1_medin$Type)
exp1_medin$Subject <- as.factor(exp1_medin$Subject)
exp1_medin$Model <- as.factor(exp1_medin$Model)

# Add adjusted ratings
exp1_pred$Item <- as.factor(exp1_pred$Item)
exp1_pred <- exp1_pred %>%
  left_join(exp1_medin %>%
              select(Item, Rating_adj), by = "Item")

exp1_pred <- exp1_pred %>%
  select(Subject, Item, Category, Type, Prediction, Rating_adj)
exp1_pred$Model <- "Cobweb"
exp1_pred$Category <- as.factor(exp1_pred$Category)
exp1_pred$Type <- as.factor(exp1_pred$Type)
exp1_pred$Subject <- as.factor(exp1_pred$Subject)
exp1_pred$Model <- as.factor(exp1_pred$Model)

exp1_data <- bind_rows(exp1_medin, exp1_pred)

exp1_data_avg <- exp1_data %>%
  group_by(Model, Item, Type, Category) %>%
  summarise(
    Rating_adj  = mean(Rating_adj, na.rm = TRUE),
    Prediction = mean(Prediction, na.rm = TRUE),
    .groups = "drop"
  )

exp1_data <- exp1_data %>%
  mutate(TypeCategory = interaction(Category, Type, sep = "-"))


corr_df_exp1 <- exp1_data_avg %>%
  group_by(Model) %>%
  summarise(
    spearman_rho = cor(
      Rating_adj,
      Prediction,
      method = "spearman",
      use = "complete.obs"
    ),
    .groups = "drop"
  )

model_labels_exp1 <- corr_df_exp1 %>%
  mutate(label = sprintf("%s (Spearman \u03c1 = %.2f)", Model, spearman_rho)) %>%
  select(Model, label) %>%
  deframe()

ggplot(exp1_data, aes(x = reorder(Item, Rating_adj), y = Prediction, shape=TypeCategory, fill=TypeCategory,
                      group = interaction(Model, Type, Category),
                      linetype = Model, label=Item)) +
  stat_summary(fun = mean, geom = "point", size=3) +
  stat_summary(fun = mean, geom = "line") +
  stat_summary(fun.data = mean_cl_boot, geom = "errorbar", width=0.15, alpha=0.7) +
  #stat_summary(fun = mean, geom = "text", position = position_nudge(x = 0.01, y = 0.02)) +
  #geom_point() +
  #geom_line() +
  #geom_text(nudge_x = 0.01, nudge_y = 0.01) +
  scale_linetype_manual(
    values = c(
    "Cobweb" = "solid",
    "RMC" = "dashed"
    ),
    labels = model_labels_exp1
  ) +
  scale_shape_manual(values = c(
    "A-Old"  = 21,  # circle (fillable)
    "A-New" = 24,   # triangle (fillable)
    "B-Old"  = 21,  # circle (fillable)
    "B-New" = 24   # triangle (fillable)
    # "Prototypes" = 13   # triangle (fillable)
    )) +
  scale_fill_manual(values = c(
    "A-Old" = "white",   # unfilled
    "B-Old" = "black",    # filled
    "A-New" = "white",   # unfilled
    "B-New" = "black"    # filled
    )) +
  labs(
    x = "Human Confidence Rating Relative to Category A",
    # y = NULL,
    y = "Model Probability of Category A",
    # title = "Cobweb Model",
    #fill = "Category",
    #shape = "Type"
    shape = "Category & Type",
    fill  = "Category & Type"
  ) +
  theme_classic() +
  guides(
    linetype = guide_legend(order = 1, ncol = 1, byrow = TRUE),
    shape    = guide_legend(order = 2, ncol = 2, byrow = TRUE),
    fill     = guide_legend(order = 2, ncol = 2, byrow = TRUE)
  ) +
  theme(
    legend.position = c(0.41, 0.64),
    legend.justification = c("right", "bottom"),
    legend.background = element_rect(fill = "white", color = "black")
  ) +
  annotate(
  "text",
  x = Inf, y = Inf,
  label = paste(corr_df$label, collapse = "\n"),
  hjust = 1.5, vjust = 2,
  size = 3.5
)



#########
# Experiment 2 from Medin and Schaffer 1978
#########
exp2_medin<- read_csv("/Users/cmaclellan3/Projects/cobweb-psych/experiments/central-tendency-effect/results/medin_exp2_observed_and_pred.csv")
exp2_errors <- read_csv("/Users/cmaclellan3/Projects/cobweb-psych/experiments/central-tendency-effect/results/exp_medin_exp2_errors.csv")
exp2_pred <- read_csv("/Users/cmaclellan3/Projects/cobweb-psych/experiments/central-tendency-effect/results/exp_medin_exp2_results.csv")

# Reorder Human ratings to relative to class A
exp2_medin[exp2_medin$Item == "5",]$Rating <- 5.2001
exp2_medin$Rating_adj <- exp2_medin$Rating
exp2_medin[exp2_medin$Category == "B",]$Rating_adj <- -1 * (exp2_medin[exp2_medin$Category == "B",]$Rating - 3.5) + 3.5
exp2_medin[exp2_medin$Category == "B",]$Prediction <- 1 - exp2_medin[exp2_medin$Category == "B",]$Prediction

# Create human observed
exp2_human <- exp2_medin
exp2_human <- exp2_human %>%
  select(Item, Category, Type, Observed, Rating_adj)
exp2_human <- exp2_human %>%
  rename(Prediction = Observed)
exp2_human[exp2_human$Category == "B",]$Prediction <- 1 - exp2_human[exp2_human$Category == "B",]$Prediction

exp2_human$Model <- "Human"
exp2_human$Subject <- "Avg"
exp2_human$Item <- as.factor(exp2_human$Item)
exp2_human$Category <- as.factor(exp2_human$Category)
exp2_human$Type <- as.factor(exp2_human$Type)
exp2_human$Subject <- as.factor(exp2_human$Subject)
exp2_human$Model <- as.factor(exp2_human$Model)

# Preprocess model pred
exp2_medin <- exp2_medin %>%
  select(Item, Category, Type, Prediction, Rating_adj)
exp2_medin$Model <- "Context"
exp2_medin$Subject <- "Avg"
exp2_medin$Item <- as.factor(exp2_medin$Item)
exp2_medin$Category <- as.factor(exp2_medin$Category)
exp2_medin$Type <- as.factor(exp2_medin$Type)
exp2_medin$Subject <- as.factor(exp2_medin$Subject)
exp2_medin$Model <- as.factor(exp2_medin$Model)

# Add adjusted ratings
exp2_pred$Item <- as.factor(exp2_pred$Item)
exp2_pred <- exp2_pred %>%
  left_join(exp2_medin %>%
              select(Item, Rating_adj), by = "Item")

exp2_pred <- exp2_pred %>%
  select(Subject, Item, Category, Type, Prediction, Rating_adj)
exp2_pred$Model <- "Cobweb"
exp2_pred$Category <- as.factor(exp2_pred$Category)
exp2_pred$Type <- as.factor(exp2_pred$Type)
exp2_pred$Subject <- as.factor(exp2_pred$Subject)
exp2_pred$Model <- as.factor(exp2_pred$Model)


exp2_data <- bind_rows(exp2_medin, exp2_pred) # , exp2_human)

exp2_data_avg <- exp2_data %>%
  group_by(Model, Item, Type, Category) %>%
  summarise(
    Rating_adj  = mean(Rating_adj, na.rm = TRUE),
    Prediction = mean(Prediction, na.rm = TRUE),
    .groups = "drop"
  )

exp2_data <- exp2_data %>%
  mutate(TypeCategory = interaction(Category, Type, sep = "-"))

corr_df_exp2 <- exp2_data_avg %>%
  group_by(Model) %>%
  summarise(
    spearman_rho = cor(
      Rating_adj,
      Prediction,
      method = "spearman",
      use = "complete.obs"
    ),
    .groups = "drop"
  )

model_labels_exp2 <- corr_df_exp2 %>%
  mutate(label = sprintf("%s (Spearman \u03c1 = %.2f)", Model, spearman_rho)) %>%
  select(Model, label) %>%
  deframe()

ggplot(exp2_data, aes(x = reorder(Item, Rating_adj), y = Prediction, shape=TypeCategory, fill=TypeCategory,
                      group = interaction(Model, Type, Category),
                      linetype = Model, label=Item)) +
  stat_summary(fun = mean, geom = "point", size=2) +
  stat_summary(fun = mean, geom = "line") +
  stat_summary(fun.data = mean_cl_boot, geom = "errorbar", width=0.15, alpha=0.7) +
  #geom_text(data = exp2_data_avg[exp2_data_avg$Model == "Cobweb",],
  #          aes(y = Prediction, label = Item),
  #          position = position_jitter(width = 0.00, height = 0.00),
  #          vjust = 2) +
  #stat_summary(fun = mean, geom = "text", position = position_nudge(x = 0.00, y = -0.04)) +
  #geom_point() +
  #geom_line() +
  #geom_text(nudge_x = 0.01, nudge_y = 0.01) +
  scale_linetype_manual(values = c(
    "Cobweb" = "solid",
    "Context" = "dashed"
    ),
    labels = model_labels_exp2
  ) +
  scale_shape_manual(values = c(
    "A-Old"  = 21,  # circle (fillable)
    "A-New" = 24,   # triangle (fillable)
    "B-Old"  = 21,  # circle (fillable)
    "B-New" = 24   # triangle (fillable)
    # "Prototypes" = 13   # triangle (fillable)
  )) +
  scale_fill_manual(values = c(
    "A-Old" = "white",   # unfilled
    "B-Old" = "black",    # filled
    "A-New" = "white",   # unfilled
    "B-New" = "black"    # filled
  )) +
  labs(
    x = "Items Ranked by Human Category A Confidence",
    # y = NULL,
    y = "Model Probability of Category A",
    # title = "Cobweb Model",
    #fill = "Category",
    #shape = "Type"
    shape = "Category & Type",
    fill  = "Category & Type"
  ) +
  theme_classic() +
  guides(
    linetype = guide_legend(order = 1, ncol = 1, byrow = TRUE),
    shape    = guide_legend(order = 2, ncol = 2, byrow = TRUE),
    fill     = guide_legend(order = 2, ncol = 2, byrow = TRUE)
  ) +
  theme(
    legend.position = c(0.43, 0.64),
    legend.justification = c("right", "bottom"),
    legend.background = element_rect(fill = "white", color = "black")
  ) +
  annotate(
    "text",
    x = Inf, y = Inf,
    label = paste(corr_df$label, collapse = "\n"),
    hjust = 1.5, vjust = 2,
    size = 3.5
  )

