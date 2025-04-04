library(reticulate) # Enables integration with Python
library(dplyr) # For data manipulation
library(pROC) # For ROC curve and AUC calculations

deephit_model <- function(train_data, test_data, epochs = 10, num_nodes = c(64L, 64L), 
                          dropout = 0, batch_size = 256L, lr = 0.001, class_weight = NULL) {
  # Validate input data: ensure that both train_data and test_data are provided and contain the 'event' column
  if (is.null(train_data) || is.null(test_data)) {
    stop("Error: train_data or test_data is NULL!")
  }
  if (nrow(train_data) == 0 || nrow(test_data) == 0) {
    stop("Error: train_data or test_data is empty!")
  }
  if (!("event" %in% colnames(train_data)) || !("event" %in% colnames(test_data))) {
    stop("Error: 'event' column is missing in train_data or test_data!")
  }
  
  # Calculate the original percentage of "yes" labels before any processing
  pct_yes <- mean(train_data$event == "yes")
  
  # Split the training data into training (80%) and validation (20%) subsets for model tuning
  set.seed(123) # For reproducibility
  train_idx <- sample(seq_len(nrow(train_data)), size = 0.8 * nrow(train_data))
  val_idx <- setdiff(seq_len(nrow(train_data)), train_idx)
  train_subset_orig <- train_data[train_idx, ]
  val_subset <- train_data[val_idx, ]
  
  # Candidate 1: Without SMOTE
  train_subset_no_smote <- train_subset_orig
  # Extract features and labels from candidate 1 training subset and validation subset
  X_train1 <- train_subset_no_smote %>% select(-event)
  X_val <- val_subset %>% select(-event)
  y_train1 <- train_subset_no_smote$event
  y_val <- val_subset$event
  # Convert feature data frames to matrices
  X_train1 <- as.matrix(X_train1)
  X_val <- as.matrix(X_val)
  # Convert labels to binary numeric values: 1 for "yes", 0 for "no"
  y_train1 <- ifelse(y_train1 == "yes", 1, 0)
  y_val_numeric <- ifelse(y_val == "yes", 1, 0)
  # Package candidate 1 training and validation data into a list for Python
  train_data_py_1 <- r_to_py(list(
    X = X_train1,
    y = y_train1,
    val_X = X_val,
    val_y = y_val_numeric,
    epochs = as.integer(epochs),
    num_nodes = num_nodes,
    dropout = dropout,
    batch_size = as.integer(batch_size),
    lr = lr
  ))
  
  if (pct_yes < 0.15) {
    # Candidate 2: With SMOTE (only used if "yes" proportion is less than 15%)
    train_subset_smote <- smote_data(train_subset_orig)
    # Extract features and labels from candidate 2 training subset and use same validation subset as candidate 1
    X_train2 <- train_subset_smote %>% select(-event)
    y_train2 <- train_subset_smote$event
    X_train2 <- as.matrix(X_train2)
    y_train2 <- ifelse(y_train2 == "yes", 1, 0)
    # Package candidate 2 training data with same validation set
    train_data_py_2 <- r_to_py(list(
      X = X_train2,
      y = y_train2,
      val_X = X_val,
      val_y = y_val_numeric,
      epochs = as.integer(epochs),
      num_nodes = num_nodes,
      dropout = dropout,
      batch_size = as.integer(batch_size),
      lr = lr
    ))
  }
  
  # Package test data for prediction
  X_test <- test_data %>% select(-event)
  y_test <- test_data$event
  X_test <- as.matrix(X_test)
  y_test_numeric <- ifelse(y_test == "yes", 1, 0)
  test_data_py <- r_to_py(list(
    X = X_test,
    y = y_test_numeric
  ))
  
  # Convert class_weight to a Python dictionary if provided
  class_weight_py <- NULL
  if (!is.null(class_weight)) {
    class_weight_py <- r_to_py(as.list(class_weight))
  }
  
  # Load the Python script that implements the DeepHit model
  source_python("deephit_model.py")
  
  # Train candidate 1 DeepHit model by calling the Python training function
  deep_model_1 <- train_deephit(train_data_py_1, class_weight = class_weight_py)
  # Predict probabilities on the validation set for candidate 1
  val_pred_prob_1 <- deep_model_1$predict_proba(train_data_py_1[["val_X"]])
  val_pred_prob_1 <- py_to_r(val_pred_prob_1)
  # Assume the second column corresponds to the probability for class "1"
  val_pred_prob_class1_1 <- val_pred_prob_1[, 2]
  # Calculate the ROC curve on the validation data for candidate 1 using continuous predicted probabilities
  roc_val_1 <- roc(response = y_val_numeric, predictor = as.vector(val_pred_prob_class1_1))
  auc_val_1 <- auc(roc_val_1)
  optimal_threshold_1 <- coords(roc_val_1, "best", ret = "threshold", best.method = "youden")
  optimal_threshold_1 <- as.numeric(optimal_threshold_1)[1]
  
  if (pct_yes < 0.15) {
    # Train candidate 2 DeepHit model by calling the Python training function
    deep_model_2 <- train_deephit(train_data_py_2, class_weight = class_weight_py)
    # Predict probabilities on the validation set for candidate 2
    val_pred_prob_2 <- deep_model_2$predict_proba(train_data_py_2[["val_X"]])
    val_pred_prob_2 <- py_to_r(val_pred_prob_2)
    # Assume the second column corresponds to the probability for class "1"
    val_pred_prob_class1_2 <- val_pred_prob_2[, 2]
    # Calculate the ROC curve on the validation data for candidate 2 using continuous predicted probabilities
    roc_val_2 <- roc(response = y_val_numeric, predictor = as.vector(val_pred_prob_class1_2))
    auc_val_2 <- auc(roc_val_2)
    optimal_threshold_2 <- coords(roc_val_2, "best", ret = "threshold", best.method = "youden")
    optimal_threshold_2 <- as.numeric(optimal_threshold_2)[1]
    
    # Decide which candidate model to use based on validation AUC
    if (auc_val_2 > auc_val_1) {
      final_model <- deep_model_2
      final_threshold <- optimal_threshold_2
      model_version <- "smote"
    }
    else {
      final_model <- deep_model_1
      final_threshold <- optimal_threshold_1
      model_version <- "no smote"
    }
  }
  else {
    final_model <- deep_model_1
    final_threshold <- optimal_threshold_1
    model_version <- "no smote"
  }
  
  # Use the selected DeepHit model to predict probabilities on the test dataset
  pred_prob <- final_model$predict_proba(test_data_py[["X"]])
  pred_prob <- py_to_r(pred_prob)
  # Assume that the second column corresponds to the probability for class "1"
  pred_prob_class1 <- pred_prob[, 2]
  # Calculate the ROC curve and AUC score for the test set using continuous predicted probabilities
  roc_curve <- roc(response = y_test_numeric, predictor = as.vector(pred_prob_class1))
  auc_score <- auc(roc_curve)
  # Plot the ROC curve with the AUC value displayed in the title
  plot(roc_curve, main = paste("ROC Curve (AUC =", round(auc_score, 4), ")"))
  # Convert predicted probabilities to binary class labels using the optimal threshold
  predictions <- ifelse(pred_prob_class1 > final_threshold, 1, 0)
  # Create a confusion matrix comparing the predicted labels to the actual test labels
  test_conf_matrix <- table(
    Predicted = factor(predictions, levels = c(0, 1)),
    Actual = factor(y_test_numeric, levels = c(0, 1))
  )
  # Calculate model performance metrics using a custom function
  metrics_dh <- calculate_model_metrics(test_conf_matrix, predictions, "DeepHit", auc_score)
  metrics_dh_dataframe <- get_dataframe("DeepHit", metrics_dh)
  return(list(metrics_dh_dataframe = metrics_dh_dataframe, metrics_dh = metrics_dh, model_version = model_version))
}

deepLearning_classification_model <- function(train_data, test_data, epochs = 10, num_nodes = c(64L, 64L), 
                                              dropout = 0, batch_size = 256L, lr = 0.001, class_weight = NULL) {
  # Check that data is provided and contains the 'event' column
  if (is.null(train_data) || is.null(test_data)) {
    stop("Error: train_data or test_data is NULL!")
  }
  if (nrow(train_data) == 0 || nrow(test_data) == 0) {
    stop("Error: train_data or test_data is empty!")
  }
  if (!("event" %in% colnames(train_data)) || !("event" %in% colnames(test_data))) {
    stop("Error: 'event' column is missing!")
  }
  
  # Calculate the original percentage of "yes" labels before any processing
  pct_yes <- mean(train_data$event == "yes")
  
  # Split training data into training (80%) and validation (20%) subsets
  set.seed(123)
  train_idx <- sample(seq_len(nrow(train_data)), size = 0.8 * nrow(train_data))
  val_idx <- setdiff(seq_len(nrow(train_data)), train_idx)
  train_subset_orig <- train_data[train_idx, ]
  val_subset <- train_data[val_idx, ]
  
  # Candidate 1: Without SMOTE
  X_train1 <- as.matrix(train_subset_orig %>% select(-event))
  X_val <- as.matrix(val_subset %>% select(-event))
  y_train1 <- ifelse(train_subset_orig$event == "yes", 1, 0)
  y_val_numeric <- ifelse(val_subset$event == "yes", 1, 0)
  
  train_data_py_1 <- r_to_py(list(
    X = X_train1,
    y = y_train1,
    val_X = X_val,
    val_y = y_val_numeric,
    epochs = as.integer(epochs),
    num_nodes = num_nodes,
    dropout = dropout,
    batch_size = as.integer(batch_size),
    lr = lr
  ))
  
  X_test <- as.matrix(test_data %>% select(-event))
  y_test <- test_data$event
  y_test_numeric <- ifelse(y_test == "yes", 1, 0)
  test_data_py <- r_to_py(list(
    X = X_test,
    y = y_test_numeric
  ))
  
  class_weight_py <- NULL
  if (!is.null(class_weight)) {
    class_weight_py <- r_to_py(as.list(class_weight))
  }
  
  source_python("deepLearning_classification_model.py")
  
  deepLearning_model_1 <- train_deepLearning(train_data_py_1, class_weight = class_weight_py)
  val_pred_prob_1 <- py_to_r(deepLearning_model_1$predict_proba(train_data_py_1[["val_X"]]))
  val_pred_prob_class1_1 <- val_pred_prob_1[, 2]
  roc_val_1 <- roc(response = y_val_numeric, predictor = as.vector(val_pred_prob_class1_1))
  auc_val_1 <- auc(roc_val_1)
  optimal_threshold_1 <- coords(roc_val_1, "best", ret = "threshold", best.method = "youden")
  optimal_threshold_1 <- as.numeric(optimal_threshold_1)[1]
  print(paste("Candidate 1 Validation AUC:", round(auc_val_1, 4)))
  
  if (pct_yes < 0.15) {
    # Candidate 2: With SMOTE
    train_subset_smote <- smote_data(train_subset_orig)
    X_train2 <- as.matrix(train_subset_smote %>% select(-event))
    y_train2 <- ifelse(train_subset_smote$event == "yes", 1, 0)
    
    train_data_py_2 <- r_to_py(list(
      X = X_train2,
      y = y_train2,
      val_X = X_val,
      val_y = y_val_numeric,
      epochs = as.integer(epochs),
      num_nodes = num_nodes,
      dropout = dropout,
      batch_size = as.integer(batch_size),
      lr = lr
    ))
    
    deepLearning_model_2 <- train_deepLearning(train_data_py_2, class_weight = class_weight_py)
    val_pred_prob_2 <- py_to_r(deepLearning_model_2$predict_proba(train_data_py_2[["val_X"]]))
    val_pred_prob_class1_2 <- val_pred_prob_2[, 2]
    roc_val_2 <- roc(response = y_val_numeric, predictor = as.vector(val_pred_prob_class1_2))
    auc_val_2 <- auc(roc_val_2)
    optimal_threshold_2 <- coords(roc_val_2, "best", ret = "threshold", best.method = "youden")
    optimal_threshold_2 <- as.numeric(optimal_threshold_2)[1]
    print(paste("Candidate 2 Validation AUC:", round(auc_val_2, 4)))
    
    if (auc_val_2 > auc_val_1) {
      final_model <- deepLearning_model_2
      final_threshold <- optimal_threshold_2
      model_version <- "smote"
    }
    else {
      final_model <- deepLearning_model_1
      final_threshold <- optimal_threshold_1
      model_version <- "no smote"
    }
  }
  else {
    final_model <- deepLearning_model_1
    final_threshold <- optimal_threshold_1
    model_version <- "no smote"
  }
  
  pred_prob <- py_to_r(final_model$predict_proba(test_data_py[["X"]]))
  pred_prob_class1 <- pred_prob[, 2]
  roc_curve <- roc(response = y_test_numeric, predictor = as.vector(pred_prob_class1))
  auc_score <- auc(roc_curve)
  plot(roc_curve, main = paste("ROC Curve (AUC =", round(auc_score, 4), ")"))
  
  predictions <- ifelse(pred_prob_class1 > final_threshold, 1, 0)
  test_conf_matrix <- table(
    Predicted = factor(predictions, levels = c(0, 1)),
    Actual = factor(y_test_numeric, levels = c(0, 1))
  )
  
  metrics_dlc <- calculate_model_metrics(test_conf_matrix, predictions, "DeepLearningClassifier", auc_score)
  metrics_dlc_dataframe <- get_dataframe("DeepLearningClassifier", metrics_dlc)
  return(list(metrics_dlc_dataframe = metrics_dlc_dataframe, metrics_dlc = metrics_dlc, model_version = model_version))
}