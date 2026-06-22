# TB Detection from Chest X-Rays — Transfer Learning with Keras
#
# Dataset: Tuberculosis (TB) Chest X-ray Database
#   URL:   https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset
#   Size:  700 TB images + 3500 Normal images (class imbalance ~1:5)
#
# Approach: Transfer learning with MobileNetV2, fine-tuned for binary classification
# ══════════════════════════════════════════════════════════════════════════════

# ── 0. Installation (run once) ───────────────────────────────────────────────
# install.packages(c("keras3", "tensorflow", "tidyverse", "pROC", "caret"))
# keras3::install_keras()   # installs TensorFlow + Keras backend


# ── 1. Data Download from Kaggle ─────────────────────────────────────────────
# Option A: Kaggle API (recommended)
#   1. Create a Kaggle account at kaggle.com
#   2. Go to Account -> Create New API Token -> downloads kaggle.json
#   3. Place kaggle.json in ~/.kaggle/ (Windows: C:/Users/<username>/.kaggle/)
#   4. Run the shell command below:
#
#   kaggle datasets download -d tawsifurrahman/tuberculosis-tb-chest-xray-dataset
#   unzip tuberculosis-tb-chest-xray-dataset.zip -d data/TB_CXR
#
# Option B: Manual download from:
#   https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset
#   Extract to:  data/TB_CXR/
#
# Expected folder structure after extraction:
#   data/TB_CXR/
#     Normal/        (3500 images: Normal-1.png ... Normal-3500.png)
#     Tuberculosis/  (700 images:  Tuberculosis-1.png ... Tuberculosis-700.png)
remotes::install_github("rstudio/tfdatasets")

# ── 2. Setup ─────────────────────────────────────────────────────────────────
library(keras3)
library(tensorflow)
library(tidyverse)
library(pROC)
library(caret)
library(reticulate)
library(tfdatasets)

use_python("C:/Users/imaposa/AppData/Local/Programs/Python/Python313/python.exe")

TF_ENABLE_ONEDNN_OPTS <- 0
set.seed(8523)
tf$random$set_seed(8523L)

data_dir   <- "Data/TB_Chest_Radiography_Database"
img_size   <- c(224L, 224L)
batch_size <- 32L


# ── 3. Explore the Raw Data ──────────────────────────────────────────────────
n_normal <- length(list.files(file.path(data_dir, "Normal")))
n_tb     <- length(list.files(file.path(data_dir, "Tuberculosis")))

cat("Normal images:       ", n_normal, "\n")
cat("Tuberculosis images: ", n_tb, "\n")
cat("Class ratio (N:TB):  ", round(n_normal / n_tb, 1), ":1\n")

total <- n_normal + n_tb
class_weight <- list(
  "0" = total / (2 * n_normal),
  "1" = total / (2 * n_tb)
)
cat("Class weights -- Normal:", round(class_weight[["0"]], 3),
    "  TB:", round(class_weight[["1"]], 3), "\n")


# ── 4. Data Pipeline ─────────────────────────────────────────────────────────
# keras3 uses image_dataset_from_directory() which returns a tf.data.Dataset.
# Rescaling and augmentation are handled as preprocessing layers in the model.
#
# Split: 70% train / 15% validation / 15% test

prepare_directories <- function(src_dir, out_dir, test_frac = 0.15, seed = 8523) {
  set.seed(seed)
  classes <- list.dirs(src_dir, recursive = FALSE, full.names = FALSE)
  for (cls in classes) {
    files     <- list.files(file.path(src_dir, cls), full.names = TRUE)
    n_test    <- round(length(files) * test_frac)
    test_idx  <- sample(seq_along(files), n_test)
    train_idx <- setdiff(seq_along(files), test_idx)
    for (split in c("train", "test")) {
      dir_path <- file.path(out_dir, split, cls)
      if (!dir.exists(dir_path)) dir.create(dir_path, recursive = TRUE)
    }
    file.copy(files[train_idx], file.path(out_dir, "train", cls))
    file.copy(files[test_idx],  file.path(out_dir, "test",  cls))
  }
  cat("Directories created at:", out_dir, "\n")
}

split_dir <- "Data/TB_CXR_split"
if (!dir.exists(split_dir)) {
  prepare_directories(data_dir, split_dir, test_frac = 0.15)
}

# Raw pixel datasets (0-255); rescaling is handled inside the model.
# Note: dataset_prefetch() wraps the dataset and drops class_names, so we
# read class_names from the raw dataset before applying prefetch.
train_raw <- image_dataset_from_directory(
  directory        = file.path(split_dir, "train"),
  labels           = "inferred",
  label_mode       = "binary",
  image_size       = img_size,
  batch_size       = batch_size,
  validation_split = 0.1765,
  subset           = "training",
  seed             = 8523L
)
cat("Class names (index order):", paste(train_raw$class_names, collapse = ", "), "\n")
# Normal = index 0, Tuberculosis = index 1
train_gen <- train_raw |> dataset_prefetch(buffer_size = tf$data$AUTOTUNE)

val_gen <- image_dataset_from_directory(
  directory        = file.path(split_dir, "train"),
  labels           = "inferred",
  label_mode       = "binary",
  image_size       = img_size,
  batch_size       = batch_size,
  validation_split = 0.1765,
  subset           = "validation",
  seed             = 8523L
) |> dataset_prefetch(buffer_size = tf$data$AUTOTUNE)

test_gen <- image_dataset_from_directory(
  directory    = file.path(split_dir, "test"),
  labels       = "inferred",
  label_mode   = "binary",
  image_size   = img_size,
  batch_size   = batch_size,
  shuffle      = FALSE
) |> dataset_prefetch(buffer_size = tf$data$AUTOTUNE)


# ── 5. Sample Images ─────────────────────────────────────────────────────────
plot_sample_images <- function(dataset, n = 6, title = "Sample Images") {
  batch  <- reticulate::iter_next(reticulate::as_iterator(dataset))
  images <- as.array(batch[[1]])
  labels <- as.array(batch[[2]])
  
  par(mfrow = c(2, 3), mar = c(1, 1, 2, 1))
  for (i in seq_len(min(n, dim(images)[1]))) {
    img <- images[i,,,] / 255
    img <- aperm(img, c(2, 1, 3))
    plot(as.raster(img), main = ifelse(labels[i] == 1, "TB", "Normal"))
  }
  par(mfrow = c(1, 1))
}

plot_sample_images(train_gen, title = "Training sample")


# ── 6. Model: MobileNetV2 Transfer Learning ──────────────────────────────────
# Strategy:
#   Phase 1 -- Feature extraction: freeze the MobileNetV2 base, train only the
#              new classification head.
#   Phase 2 -- Fine-tuning: unfreeze top layers with a very low learning rate.
#
# Preprocessing layers are baked into the model:
#   - layer_rescaling: normalises pixels from [0,255] to [0,1]
#   - Random augmentation layers: active during training, no-ops at inference

base_model <- application_mobilenet_v2(
  input_shape = c(img_size, 3L),
  include_top = FALSE,
  weights     = "imagenet"
)
freeze_weights(base_model)
# preprocessing -> frozen base -> classification head
inputs <- layer_input(shape = c(img_size, 3L))

preprocessed <- inputs |>
  layer_rescaling(scale = 1/255) |>
  layer_random_flip("horizontal") |>
  layer_random_rotation(factor = 0.028) |>
  layer_random_zoom(height_factor = 0.1) |>
  layer_random_translation(height_factor = 0.05, width_factor = 0.05)

# training = FALSE keeps MobileNetV2 batch norm in inference mode (Phase 1)
base_out <- base_model(preprocessed, training = FALSE)

output <- base_out |>
  layer_global_average_pooling_2d() |>
  layer_dropout(rate = 0.3) |>
  layer_dense(units = 128, activation = "relu") |>
  layer_batch_normalization() |>
  layer_dropout(rate = 0.2) |>
  layer_dense(units = 1, activation = "sigmoid")

model <- keras_model(inputs = inputs, outputs = output)

model |> compile(
  optimizer = optimizer_adam(learning_rate = 1e-3),
  loss      = "binary_crossentropy",
  metrics   = list("accuracy", metric_auc(name = "auc"))
)

n_trainable <- sum(sapply(model$trainable_weights, function(w) prod(w$shape$as_list())))
cat("Trainable parameters: ", format(n_trainable, big.mark = ","), "\n")


# ── 7. Callbacks ─────────────────────────────────────────────────────────────
if (!dir.exists("models")) dir.create("models")

callbacks_list <- list(
  callback_early_stopping(
    monitor              = "val_auc",
    patience             = 8,
    mode                 = "max",
    restore_best_weights = TRUE
  ),
  callback_reduce_lr_on_plateau(
    monitor  = "val_loss",
    factor   = 0.5,
    patience = 4,
    min_lr   = 1e-7,
    verbose  = 1
  ),
  callback_model_checkpoint(
    filepath       = "models/tb_mobilenetv2_phase1.keras",
    monitor        = "val_auc",
    save_best_only = TRUE,
    mode           = "max"
  )
)


# ── 8. Phase 1 Training: Feature Extraction ──────────────────────────────────
cat("\n-- Phase 1: Feature Extraction --\n")

history1 <- model |> fit(
  train_gen,
  epochs          = 30,
  validation_data = val_gen,
  class_weight    = class_weight,
  callbacks       = callbacks_list
)

#history1


# ── 9. Phase 2 Training: Fine-Tuning ─────────────────────────────────────────
n_layers   <- length(base_model$layers)
freeze_idx <- round(n_layers * 0.7)

for (i in seq_len(freeze_idx)) {
  lyr <- base_model$layers[[i]]
  lyr$trainable <- FALSE
}
for (i in seq(freeze_idx + 1, n_layers)) {
  lyr <- base_model$layers[[i]]
  lyr$trainable <- TRUE
}

model |> compile(
  optimizer = optimizer_adam(learning_rate = 1e-5),
  loss      = "binary_crossentropy",
  metrics   = list("accuracy", metric_auc(name = "auc"))
)

n_trainable2 <- sum(sapply(model$trainable_weights, function(w) prod(w$shape$as_list())))
cat("Trainable parameters (Phase 2): ", format(n_trainable2, big.mark = ","), "\n")

callbacks_list[[3]] <- callback_model_checkpoint(
  filepath       = "models/tb_mobilenetv2_final.keras",
  monitor        = "val_auc",
  save_best_only = TRUE,
  mode           = "max"
)

cat("\n-- Phase 2: Fine-Tuning --\n")

history2 <- model |> fit(
  train_gen,
  epochs          = 30,
  validation_data = val_gen,
  class_weight    = class_weight,
  callbacks       = callbacks_list
)


# ── 10. Training History Plots ───────────────────────────────────────────────
plot_history <- function(h1, h2 = NULL, metric = "loss") {
  df1 <- data.frame(
    epoch      = seq_along(h1$metrics[[metric]]),
    train      = h1$metrics[[metric]],
    validation = h1$metrics[[paste0("val_", metric)]],
    phase      = "Phase 1"
  )
  df <- df1
  if (!is.null(h2)) {
    df2 <- data.frame(
      epoch      = seq_along(h2$metrics[[metric]]) + max(df1$epoch),
      train      = h2$metrics[[metric]],
      validation = h2$metrics[[paste0("val_", metric)]],
      phase      = "Phase 2"
    )
    df <- bind_rows(df1, df2)
  }
  df |>
    pivot_longer(c(train, validation), names_to = "split", values_to = "value") |>
    ggplot(aes(epoch, value, color = split, linetype = phase)) +
    geom_line(linewidth = 0.9) +
    geom_vline(xintercept = max(df1$epoch) + 0.5, linetype = "dotted", color = "grey40") +
    labs(title = paste("Training History:", toupper(metric)),
         x = "Epoch", y = metric, color = NULL, linetype = NULL) +
    theme_minimal(base_size = 12) +
    scale_color_manual(values = c(train = "steelblue", validation = "firebrick"))
}

print(plot_history(history1, history2, metric = "loss"))
print(plot_history(history1, history2, metric = "auc"))


# ── 11. Test Set Evaluation ───────────────────────────────────────────────────
cat("\n-- Test Set Evaluation --\n")

best_model <- load_model(file.path("models", "tb_mobilenetv2_final.keras"))

# Extract true labels by iterating the unshuffled test dataset
true_labels <- unlist(lapply(
  reticulate::iterate(test_gen),
  function(batch) as.integer(as.array(batch[[2]]))
))

probs <- predict(best_model, test_gen) |> as.vector()

# ROC and AUC
roc_obj <- roc(true_labels, probs, levels = c(0, 1), direction = "<")
auc_val <- auc(roc_obj)
cat("AUC:", round(auc_val, 4), "\n")

# Youden's J optimal threshold
best_thresh <- coords(roc_obj, "best", ret = "threshold",
                      best.method = "youden")$threshold
cat("Optimal threshold (Youden's J):", round(best_thresh, 4), "\n")

pred_classes <- as.integer(probs >= best_thresh)

cm <- confusionMatrix(
  factor(pred_classes, levels = c(0, 1), labels = c("Normal", "TB")),
  factor(true_labels,  levels = c(0, 1), labels = c("Normal", "TB")),
  positive = "TB"
)
print(cm)

sensitivity <- cm$byClass["Sensitivity"]
specificity <- cm$byClass["Specificity"]
ppv         <- cm$byClass["Pos Pred Value"]
npv         <- cm$byClass["Neg Pred Value"]
f1          <- cm$byClass["F1"]

cat("\n-- Clinical Performance Summary --\n")
cat(sprintf("Sensitivity (TB recall):  %.1f%%\n", sensitivity * 100))
cat(sprintf("Specificity:              %.1f%%\n", specificity * 100))
cat(sprintf("PPV:                      %.1f%%\n", ppv * 100))
cat(sprintf("NPV:                      %.1f%%\n", npv * 100))
cat(sprintf("F1 Score:                 %.4f\n",   f1))
cat(sprintf("AUC:                      %.4f\n",   auc_val))


# ── 12. ROC Curve ─────────────────────────────────────────────────────────────
roc_df <- data.frame(
  fpr = 1 - roc_obj$specificities,
  tpr = roc_obj$sensitivities
)

ggplot(roc_df, aes(fpr, tpr)) +
  geom_line(color = "steelblue", linewidth = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey50") +
  geom_point(
    data = data.frame(
      fpr = 1 - coords(roc_obj, "best", ret = "specificity")$specificity,
      tpr = coords(roc_obj, "best", ret = "sensitivity")$sensitivity
    ),
    color = "firebrick", size = 3
  ) +
  annotate("text", x = 0.6, y = 0.2,
           label = paste0("AUC = ", round(auc_val, 3)),
           size = 5, color = "steelblue") +
  labs(title = "ROC Curve: TB vs Normal",
       x = "False Positive Rate (1 - Specificity)",
       y = "True Positive Rate (Sensitivity)") +
  theme_minimal(base_size = 12)


# ── 13. Precision-Recall Curve ────────────────────────────────────────────────
pr_coords <- coords(roc_obj, "all", ret = c("threshold", "precision", "recall"),
                    transpose = FALSE)
pr_coords <- pr_coords[is.finite(pr_coords$precision), ]

ggplot(pr_coords, aes(recall, precision)) +
  geom_line(color = "darkgreen", linewidth = 1) +
  geom_point(
    data = pr_coords[which.min(abs(pr_coords$threshold - best_thresh)), ],
    color = "firebrick", size = 3
  ) +
  labs(title = "Precision-Recall Curve: TB Detection",
       x = "Recall (Sensitivity)", y = "Precision (PPV)") +
  theme_minimal(base_size = 12)


# ── 14. Save Model ────────────────────────────────────────────────────────────
save_model(best_model, "models/tb_cxr_final.keras")
cat("Model saved to models/tb_cxr_final.keras\n")


# ── 15. Notes on Clinical Use ─────────────────────────────────────────────────
# - In a TB screening context, sensitivity takes priority over specificity:
#   missing a TB case (false negative) is more costly than a false alarm.
#   The Youden threshold balances both, but you may wish to lower it to
#   boost sensitivity, accepting more false positives.
#
# - This model was trained on 700 TB + 3500 Normal images. Performance may
#   not generalize to other X-ray equipment, populations, or TB presentations.
#
# - Class imbalance (1:5) was addressed via class_weight. The PR curve is
#   more informative than the ROC curve under imbalance.


