# Load required libraries
library(tidyverse)
library(readr)
library(broom)
library(dunn.test)

# Load data
results <- read_csv("ablation_results/ablation_summary.csv")

# Display data structure
glimpse(results)

################################################################################
# 1. DESCRIPTIVE STATISTICS BY GROUP
################################################################################

# Summary statistics per group
group_summary <- results %>%
  group_by(group) %>%
  summarise(
    n = n(),
    mean_test_acc = mean(test_acc),
    sd_test_acc = sd(test_acc),
    mean_test_f1 = mean(test_f1),
    sd_test_f1 = sd(test_f1),
    .groups = 'drop'
  )

print(group_summary)

################################################################################
# 2. KRUSKAL-WALLIS TEST (Non-parametric ANOVA)
# H0: All groups have the same median performance
################################################################################

# Test for test_acc
kw_acc <- kruskal.test(test_acc ~ group, data = results)
print("Kruskal-Wallis Test for Test Accuracy:")
print(kw_acc)

# Test for test_f1
kw_f1 <- kruskal.test(test_f1 ~ group, data = results)
print("Kruskal-Wallis Test for Test F1:")
print(kw_f1)

################################################################################
# 3. POST-HOC PAIRWISE COMPARISONS (Dunn's Test with Bonferroni correction)
# Only if Kruskal-Wallis is significant (p < 0.05)
################################################################################

if (kw_acc$p.value < 0.05) {
  print("\n=== POST-HOC: Pairwise comparisons for Test Accuracy ===")
  dunn_acc <- dunn.test(results$test_acc, results$group, 
                        method="bonferroni", kw=TRUE, label=TRUE)
} else {
  print("Kruskal-Wallis not significant for test_acc - no post-hoc tests needed")
}

if (kw_f1$p.value < 0.05) {
  print("\n=== POST-HOC: Pairwise comparisons for Test F1 ===")
  dunn_f1 <- dunn.test(results$test_f1, results$group, 
                       method="bonferroni", kw=TRUE, label=TRUE)
} else {
  print("Kruskal-Wallis not significant for test_f1 - no post-hoc tests needed")
}

################################################################################
# 4. WITHIN-GROUP COMPARISON TO BASELINE
# Compare each experiment to baseline within its group
################################################################################

# Identify baselines (assuming description contains "baseline")
results <- results %>%
  mutate(is_baseline = grepl("baseline", description, ignore.case = TRUE))

# For each group, compare experiments to baseline using Wilcoxon signed-rank test
within_group_tests <- results %>%
  group_by(group) %>%
  filter(sum(is_baseline) > 0) %>%  # Only groups with a baseline
  summarise(
    baseline_acc = test_acc[is_baseline][1],
    wilcox_p_value = if(n() > 1) {
      wilcox.test(test_acc, mu = baseline_acc[1], alternative = "two.sided")$p.value
    } else {
      NA
    },
    .groups = 'drop'
  )

print("\n=== Within-Group Comparisons to Baseline (Wilcoxon) ===")
print(within_group_tests)

################################################################################
# 5. EFFECT SIZE (Cohen's d for pairwise comparisons)
################################################################################

# Function to calculate Cohen's d
cohens_d <- function(x1, x2) {
  n1 <- length(x1)
  n2 <- length(x2)
  pooled_sd <- sqrt(((n1-1)*sd(x1)^2 + (n2-1)*sd(x2)^2) / (n1+n2-2))
  (mean(x1) - mean(x2)) / pooled_sd
}

# Compare top 2 groups (by mean test_acc)
top_groups <- group_summary %>%
  arrange(desc(mean_test_acc)) %>%
  head(2) %>%
  pull(group)

if (length(top_groups) == 2) {
  g1_data <- results %>% filter(group == top_groups[1]) %>% pull(test_acc)
  g2_data <- results %>% filter(group == top_groups[2]) %>% pull(test_acc)
  
  effect_size <- cohens_d(g1_data, g2_data)
  
  cat(sprintf("\nCohen's d between %s and %s: %.3f\n", 
              top_groups[1], top_groups[2], effect_size))
  cat("Interpretation: |d| < 0.2 (small), 0.2-0.5 (medium), 0.5-0.8 (large), > 0.8 (very large)\n")
}

################################################################################
# 6. BEST EXPERIMENT IDENTIFICATION
################################################################################

best_exp <- results %>%
  arrange(desc(test_acc)) %>%
  head(1) %>%
  select(exp_id, group, description, test_acc, test_f1, num_params)

print("\n=== Best Overall Experiment ===")
print(best_exp)

# Best per group
best_per_group <- results %>%
  group_by(group) %>%
  slice_max(test_acc, n = 1) %>%
  select(group, exp_id, description, test_acc, test_f1, num_params)

print("\n=== Best Experiment per Group ===")
print(best_per_group)

################################################################################
# 7. VISUALIZATION: BOXPLOT
################################################################################

# Boxplot of test accuracy by group
ggplot(results, aes(x = reorder(group, test_acc, FUN = median), 
                    y = test_acc, fill = group)) +
  geom_boxplot(alpha = 0.7) +
  geom_jitter(width = 0.2, alpha = 0.5) +
  labs(title = "Test Accuracy Distribution by Ablation Group",
       x = "Group", y = "Test Accuracy") +
  theme_minimal() +
  theme(legend.position = "none")

ggsave("ablation_results/boxplot_test_acc.png", width = 10, height = 6, dpi = 300)

################################################################################
# 8. EXPORT RESULTS
################################################################################

# Create results summary
results_summary <- tibble(
  test = c("Kruskal-Wallis (Acc)", "Kruskal-Wallis (F1)"),
  statistic = c(kw_acc$statistic, kw_f1$statistic),
  p_value = c(kw_acc$p.value, kw_f1$p.value),
  significant = c(kw_acc$p.value < 0.05, kw_f1$p.value < 0.05)
)

write_csv(results_summary, "ablation_results/statistical_tests_summary.csv")
write_csv(group_summary, "ablation_results/group_summary_stats.csv")

cat("\n=== Analysis Complete ===\n")
cat("Results saved to:\n")
cat("  - ablation_results/statistical_tests_summary.csv\n")
cat("  - ablation_results/group_summary_stats.csv\n")
cat("  - ablation_results/boxplot_test_acc.png\n")