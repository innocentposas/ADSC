# =============================================================================
# Simple ANN vs Logistic Regression - Binary Outcome Prediction
# With 10-Fold Cross-Validation
# =============================================================================
# Dataset: Pima Indians Diabetes (built-in via mlbench)
# Outcome: diabetes (pos/neg)
# =============================================================================

# Install packages if needed
# install.packages(c("mlbench", "neuralnet", "caret", "tidyverse", "pROC"))

library(mlbench)
library(neuralnet)
library(caret)
library(tidyverse)
library(pROC)

# -----------------------------------------------------------------------------
# 1. Load & Prepare Data
# -----------------------------------------------------------------------------
data("PimaIndiansDiabetes")
df <- PimaIndiansDiabetes

# Convert outcome to binary numeric (required by neuralnet)
df$diabetes_bin <- ifelse(df$diabetes == "pos", 1, 0)
df$diabetes <- NULL

# Check class balance
table(df$diabetes_bin)

# Normalize numeric predictors to [0, 1] (important for ANN)
normalize <- function(x) (x - min(x)) / (max(x) - min(x))

df_norm <- df |>
  mutate(across(-diabetes_bin, normalize))

# -----------------------------------------------------------------------------
# 2. Train / Test Split (70/30)
# -----------------------------------------------------------------------------
set.seed(4821)
train_idx <- createDataPartition(df_norm$diabetes_bin, p = 0.7, list = FALSE)
train <- df_norm[ train_idx, ]
test  <- df_norm[-train_idx, ]

cat("Training rows:", nrow(train), "| Test rows:", nrow(test), "\n")

# -----------------------------------------------------------------------------
# 3. Logistic Regression
# -----------------------------------------------------------------------------
log_model <- glm(diabetes_bin ~ ., data = train, family = binomial)
summary(log_model)

# Predict on test set
log_probs <- predict(log_model, newdata = test, type = "response")
log_pred  <- ifelse(log_probs >= 0.5, 1, 0) |> factor(levels = c(0, 1))
actual    <- factor(test$diabetes_bin, levels = c(0, 1))

log_cm <- confusionMatrix(log_pred, actual, positive = "1")
cat("\n--- Logistic Regression ---\n")
print(log_cm$table)
cat("Accuracy:", round(log_cm$overall["Accuracy"], 4), "\n")
cat("Sensitivity:", round(log_cm$byClass["Sensitivity"], 4), "\n")
cat("Specificity:", round(log_cm$byClass["Specificity"], 4), "\n")

# -----------------------------------------------------------------------------
# 4. Artificial Neural Network (1 hidden layer, 5 nodes)
# -----------------------------------------------------------------------------
# Build formula: all predictors -> outcome
predictors <- setdiff(names(train), "diabetes_bin")
ann_formula <- as.formula(
  paste("diabetes_bin ~", paste(predictors, collapse = " + "))
)

set.seed(4821)
ann_model <- neuralnet(
  ann_formula,
  data       = train,
  hidden     = 5,           # 1 hidden layer with 5 neurons
  linear.output = FALSE,    # FALSE for classification (sigmoid activation)
  threshold  = 0.01,
  stepmax    = 1e6
)

# Plot the network architecture
plot(ann_model, rep = "best")

# Predict on test set
ann_raw  <- predict(ann_model, newdata = test)
ann_prob <- ann_raw[, 1]
ann_pred <- ifelse(ann_prob >= 0.5, 1, 0) |> factor(levels = c(0, 1))

ann_cm <- confusionMatrix(ann_pred, actual, positive = "1")
cat("\n--- Artificial Neural Network (hidden = 5) ---\n")
print(ann_cm$table)
cat("Accuracy:", round(ann_cm$overall["Accuracy"], 4), "\n")
cat("Sensitivity:", round(ann_cm$byClass["Sensitivity"], 4), "\n")
cat("Specificity:", round(ann_cm$byClass["Specificity"], 4), "\n")

# -----------------------------------------------------------------------------
# 5. Side-by-side Comparison
# -----------------------------------------------------------------------------
comparison <- tibble(
  Model       = c("Logistic Regression", "ANN (5 nodes)"),
  Accuracy    = c(log_cm$overall["Accuracy"],    ann_cm$overall["Accuracy"]),
  Sensitivity = c(log_cm$byClass["Sensitivity"], ann_cm$byClass["Sensitivity"]),
  Specificity = c(log_cm$byClass["Specificity"], ann_cm$byClass["Specificity"])
) |>
  mutate(across(where(is.numeric), ~ round(.x, 4)))

cat("\n===== Model Comparison =====\n")
print(comparison)

# -----------------------------------------------------------------------------
# 6. Visualise Predicted Probabilities
# -----------------------------------------------------------------------------
results <- tibble(
  Actual      = as.numeric(as.character(actual)),
  Logistic    = log_probs,
  ANN         = ann_prob
) |>
  pivot_longer(cols = c(Logistic, ANN),
               names_to = "Model", values_to = "Probability")

ggplot(results, aes(x = Probability, fill = factor(Actual))) +
  geom_histogram(bins = 30, alpha = 0.7, position = "identity") +
  facet_wrap(~Model) +
  scale_fill_manual(values = c("steelblue", "tomato"),
                    labels = c("No Diabetes", "Diabetes"),
                    name = "Actual") +
  labs(
    title = "Predicted Probabilities: Logistic Regression vs ANN",
    x     = "Predicted Probability",
    y     = "Count"
  ) +
  theme_minimal()

# =============================================================================
# 7. ROC Curves & AUC — Holdout Test Set
# =============================================================================
# pROC works with predicted probabilities (not class labels), so no thresholding
# is needed here. Higher AUC = better discrimination across all thresholds.

roc_log <- roc(test$diabetes_bin, log_probs,  quiet = TRUE)
roc_ann <- roc(test$diabetes_bin, ann_prob,   quiet = TRUE)

cat("\n--- Test-Set AUC ---\n")
cat("Logistic Regression AUC:", round(auc(roc_log), 4), "\n")
cat("ANN (5 nodes) AUC:      ", round(auc(roc_ann), 4), "\n")

# Statistical test: are the two AUCs significantly different?
roc_test <- roc.test(roc_log, roc_ann, method = "delong")
cat("\nDeLong test (H0: AUCs are equal):\n")
cat("  D =", round(roc_test$statistic, 3),
    "| p-value =", round(roc_test$p.value, 4), "\n")

# Build tidy data for plotting
roc_df <- bind_rows(
  tibble(
    Model       = paste0("Logistic Regression (AUC = ", round(auc(roc_log), 3), ")"),
    Specificity = roc_log$specificities,
    Sensitivity = roc_log$sensitivities
  ),
  tibble(
    Model       = paste0("ANN 5 nodes (AUC = ", round(auc(roc_ann), 3), ")"),
    Specificity = roc_ann$specificities,
    Sensitivity = roc_ann$sensitivities
  )
)

ggplot(roc_df, aes(x = 1 - Specificity, y = Sensitivity, colour = Model)) +
  geom_line(linewidth = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "grey50") +
  labs(
    title    = "ROC Curves — Holdout Test Set",
    x        = "1 - Specificity (False Positive Rate)",
    y        = "Sensitivity (True Positive Rate)",
    colour   = NULL
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")

# =============================================================================
# 8. 10-Fold Cross-Validation (full dataset, stratified)
# =============================================================================
# Manual CV is used because neuralnet is not natively supported by caret.
# Both models use the same folds for a fair comparison.

set.seed(4821)
k       <- 10
folds   <- createFolds(df_norm$diabetes_bin, k = k, list = TRUE, returnTrain = FALSE)

# Storage: one row per fold (includes AUC)
cv_results <- tibble(
  Fold        = integer(),
  Model       = character(),
  Accuracy    = double(),
  Sensitivity = double(),
  Specificity = double(),
  AUC         = double()
)

cat("\nRunning 10-fold CV — please wait...\n")

for (i in seq_along(folds)) {

  val_idx   <- folds[[i]]
  cv_train  <- df_norm[-val_idx, ]
  cv_test   <- df_norm[ val_idx, ]
  cv_actual <- factor(cv_test$diabetes_bin, levels = c(0, 1))

  # ---- Logistic Regression --------------------------------------------------
  log_cv   <- glm(diabetes_bin ~ ., data = cv_train, family = binomial)
  log_p    <- predict(log_cv, newdata = cv_test, type = "response")
  log_cls  <- factor(ifelse(log_p >= 0.5, 1, 0), levels = c(0, 1))
  log_met  <- confusionMatrix(log_cls, cv_actual, positive = "1")

  cv_results <- cv_results |> add_row(
    Fold        = i,
    Model       = "Logistic Regression",
    Accuracy    = log_met$overall["Accuracy"],
    Sensitivity = log_met$byClass["Sensitivity"],
    Specificity = log_met$byClass["Specificity"],
    AUC         = as.numeric(auc(roc(cv_test$diabetes_bin, log_p, quiet = TRUE)))
  )

  # ---- ANN (1 hidden layer, 5 nodes) ----------------------------------------
  ann_cv  <- neuralnet(
    ann_formula,
    data          = cv_train,
    hidden        = 5,
    linear.output = FALSE,
    threshold     = 0.01,
    stepmax       = 1e6
  )
  ann_p   <- predict(ann_cv, newdata = cv_test)[, 1]
  ann_cls <- factor(ifelse(ann_p >= 0.5, 1, 0), levels = c(0, 1))
  ann_met <- confusionMatrix(ann_cls, cv_actual, positive = "1")

  cv_results <- cv_results |> add_row(
    Fold        = i,
    Model       = "ANN (5 nodes)",
    Accuracy    = ann_met$overall["Accuracy"],
    Sensitivity = ann_met$byClass["Sensitivity"],
    Specificity = ann_met$byClass["Specificity"],
    AUC         = as.numeric(auc(roc(cv_test$diabetes_bin, ann_p, quiet = TRUE)))
  )

  cat(sprintf("  Fold %2d complete\n", i))
}

# -----------------------------------------------------------------------------
# 9. Summarise CV Results (Mean ± SD across folds)
# -----------------------------------------------------------------------------
cv_summary <- cv_results |>
  group_by(Model) |>
  summarise(
    across(
      c(Accuracy, Sensitivity, Specificity, AUC),
      list(Mean = mean, SD = sd),
      .names = "{.col}_{.fn}"
    ),
    .groups = "drop"
  ) |>
  mutate(across(where(is.numeric), ~ round(.x, 4)))

cat("\n===== 10-Fold CV Summary (Mean ± SD) =====\n")
print(cv_summary)

# Formatted table: Mean (SD)
cv_pretty <- cv_summary |>
  transmute(
    Model,
    Accuracy    = paste0(Accuracy_Mean,    " (±", Accuracy_SD,    ")"),
    Sensitivity = paste0(Sensitivity_Mean, " (±", Sensitivity_SD, ")"),
    Specificity = paste0(Specificity_Mean, " (±", Specificity_SD, ")"),
    AUC         = paste0(AUC_Mean,         " (±", AUC_SD,         ")")
  )

cat("\n===== Formatted CV Summary =====\n")
print(cv_pretty)

# -----------------------------------------------------------------------------
# 10. Visualise CV Performance Per Fold (including AUC)
# -----------------------------------------------------------------------------
cv_results |>
  pivot_longer(cols = c(Accuracy, Sensitivity, Specificity, AUC),
               names_to = "Metric", values_to = "Value") |>
  ggplot(aes(x = factor(Fold), y = Value, colour = Model, group = Model)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2) +
  facet_wrap(~Metric, ncol = 1, scales = "free_y") +
  labs(
    title  = "10-Fold CV Performance by Fold",
    x      = "Fold",
    y      = "Metric Value",
    colour = "Model"
  ) +
  theme_minimal() +
  theme(legend.position = "top")

# -----------------------------------------------------------------------------
# 11. Boxplot: Distribution of Metrics Across Folds (including AUC)
# -----------------------------------------------------------------------------
cv_results |>
  pivot_longer(cols = c(Accuracy, Sensitivity, Specificity, AUC),
               names_to = "Metric", values_to = "Value") |>
  ggplot(aes(x = Model, y = Value, fill = Model)) +
  geom_boxplot(alpha = 0.6, outlier.shape = 21) +
  geom_jitter(width = 0.1, size = 1.5, alpha = 0.7) +
  facet_wrap(~Metric) +
  labs(
    title = "CV Performance Distribution: Logistic Regression vs ANN",
    x     = NULL,
    y     = "Metric Value"
  ) +
  theme_minimal() +
  theme(legend.position = "none",
        axis.text.x = element_text(angle = 15, hjust = 1))
