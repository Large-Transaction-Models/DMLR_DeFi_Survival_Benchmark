library(data.table)
library(dplyr)
library(glmnet)
library(rpart)
library(caret)
library(parallel)
library(xgboost)
library(pROC)

source("~/KDD_DeFi_Survival_Dataset_And_Benchmark/Classification/data_processing.R")

logistic_regression <- function(train_data, test_data) {
  # library(data.table)
  # library(caret)
  # library(pROC)
  
  # Calculate the proportion of "yes" labels in the original train_data
  pct_yes <- mean(train_data$event == "yes")
  print(pct_yes)
  
  # Convert train_data and test_data to data.table format for faster processing
  setDT(train_data)
  setDT(test_data)
  train_data[, event := ifelse(event == "yes", 1, 0)]
  test_data[, event := ifelse(event == "yes", 1, 0)]
  
  # Partition the train_data into a training subset (80%) and a validation subset (20%)
  # The validation subset will be used for threshold tuning
  set.seed(123)
  trainIndex <- createDataPartition(train_data$event, p = 0.8, list = FALSE)
  train_set <- train_data[trainIndex, ]
  validation_set <- train_data[-trainIndex, ]
  
  if (pct_yes >= 0.15) {
    # If the proportion of "yes" labels is at least 15% use the no_smote version directly
    logistic_regression_model_no_smote <- glm(event ~ ., data = train_set, family = binomial)
    val_probs_no_smote <- predict(logistic_regression_model_no_smote, newdata = validation_set, type = "response")
    roc_val_no_smote <- roc(response = validation_set$event, predictor = val_probs_no_smote)
    auc_no_smote <- auc(roc_val_no_smote)
    optimal_threshold_no_smote <- coords(roc_val_no_smote, "best", ret = "threshold", best.method = "youden")
    optimal_threshold_no_smote <- as.numeric(optimal_threshold_no_smote)[1]
    final_model <- logistic_regression_model_no_smote
    optimal_threshold <- optimal_threshold_no_smote
    model_version <- "no smote"
  }
  else {
    # If the proportion of "yes" labels is less than 15% perform a comparison between no_smote and smote versions
    # Candidate 1: Without SMOTE
    logistic_regression_model_no_smote <- glm(event ~ ., data = train_set, family = binomial)
    val_probs_no_smote <- predict(logistic_regression_model_no_smote, newdata = validation_set, type = "response")
    roc_val_no_smote <- roc(response = validation_set$event, predictor = val_probs_no_smote)
    auc_no_smote <- auc(roc_val_no_smote)
    optimal_threshold_no_smote <- coords(roc_val_no_smote, "best", ret = "threshold", best.method = "youden")
    optimal_threshold_no_smote <- as.numeric(optimal_threshold_no_smote)[1]
    
    # Candidate 2: With SMOTE
    train_set_smote <- smote_data(train_set)
    logistic_regression_model_smote <- glm(event ~ ., data = train_set_smote, family = binomial)
    val_probs_smote <- predict(logistic_regression_model_smote, newdata = validation_set, type = "response")
    roc_val_smote <- roc(response = validation_set$event, predictor = val_probs_smote)
    auc_smote <- auc(roc_val_smote)
    optimal_threshold_smote <- coords(roc_val_smote, "best", ret = "threshold", best.method = "youden")
    optimal_threshold_smote <- as.numeric(optimal_threshold_smote)[1]
    
    # Decide which model to use based on validation AUC
    if (auc_smote > auc_no_smote) {
      final_model <- logistic_regression_model_smote
      optimal_threshold <- optimal_threshold_smote
      model_version <- "smote"
    }
    else {
      final_model <- logistic_regression_model_no_smote
      optimal_threshold <- optimal_threshold_no_smote
      model_version <- "no smote"
    }
  }
  
  # Predict probabilities on the test set using the selected model
  test_probs <- predict(final_model, newdata = test_data, type = "response")
  
  # Convert probabilities to binary predictions ("yes" or "no") using the optimal threshold
  test_pred <- ifelse(test_probs > optimal_threshold, "yes", "no")
  
  # Generate a confusion matrix for the test set
  test_conf_matrix <- table(Predicted = test_pred, Actual = test_data$event)
  
  # Compute the ROC curve and AUC for the test set predictions
  roc_curve <- roc(response = test_data$event, predictor = test_probs)
  auc_score <- auc(roc_curve)
  
  # Plot the ROC curve with the AUC value displayed in the title
  plot(roc_curve, main = paste("ROC Curve (AUC =", round(auc_score, 4), ")"))
  
  # Evaluate test set performance using the custom metrics function
  metrics_lr <- calculate_model_metrics(test_conf_matrix, test_probs, "Logistic Regression", auc_score)
  
  # Create a dataframe containing the performance metrics for reporting
  metrics_lr_dataframe <- get_dataframe("Logistic Regression", metrics_lr)
  
  return(list(metrics_lr_dataframe = metrics_lr_dataframe, metrics_lr = metrics_lr, model_version = model_version))
}

decision_tree <- function(train_data, test_data) {
  # library(rpart)
  # library(caret)
  # library(pROC)
  
  # Calculate the original percentage of "yes" labels before any processing
  pct_yes <- mean(train_data$event == "yes")
  
  # Split train_data into a training set (80%) and a validation set (20%) to evaluate model performance
  set.seed(123) # Ensure reproducibility
  trainIndex <- createDataPartition(train_data$event, p = 0.8, list = FALSE)
  train_set <- train_data[trainIndex, ]
  validation_set <- train_data[-trainIndex, ]
  
  # Candidate 1: Without SMOTE
  decision_tree_model_no_smote <- rpart(
    event ~ .,
    data = train_set,
    method = "class",
    control = rpart.control(
      cp = 0.01,
      maxdepth = 30,
      minsplit = 20
    )
  )
  val_probs_no_smote <- predict(decision_tree_model_no_smote, validation_set, type = "prob")[, "yes"]
  roc_val_no_smote <- roc(response = validation_set$event, predictor = val_probs_no_smote)
  auc_no_smote <- auc(roc_val_no_smote)
  
  if (pct_yes < 0.15) {
    # Candidate 2: With SMOTE
    train_set_smote <- smote_data(train_set)
    decision_tree_model_smote <- rpart(
      event ~ .,
      data = train_set_smote,
      method = "class",
      control = rpart.control(
        cp = 0.01,
        maxdepth = 30,
        minsplit = 20
      )
    )
    val_probs_smote <- predict(decision_tree_model_smote, validation_set, type = "prob")[, "yes"]
    roc_val_smote <- roc(response = validation_set$event, predictor = val_probs_smote)
    auc_smote <- auc(roc_val_smote)
    
    if (auc_smote > auc_no_smote) {
      final_model <- decision_tree_model_smote
      model_version <- "smote"
    }
    else {
      final_model <- decision_tree_model_no_smote
      model_version <- "no smote"
    }
  }
  else {
    final_model <- decision_tree_model_no_smote
    model_version <- "no smote"
  }
  
  # Predict the class labels on the test dataset using the final model
  predict_probabilities_dt <- predict(final_model, test_data, type = "class")
  test_conf_matrix <- table(Predicted = predict_probabilities_dt, Actual = test_data$event)
  
  # Obtain continuous probabilities for the positive class on test data
  test_probs_cont <- predict(final_model, test_data, type = "prob")[, "yes"]
  roc_curve <- roc(response = test_data$event, predictor = test_probs_cont)
  auc_score <- auc(roc_curve)
  
  # Plot the ROC curve with the AUC value in the title
  plot(roc_curve, main = paste("ROC Curve (AUC =", round(auc_score, 4), ")"))
  
  # Evaluate test performance using custom metrics
  metrics_dt <- calculate_model_metrics(test_conf_matrix, predict_probabilities_dt, "Decision Tree", auc_score)
  
  # Create a dataframe with the desired structure containing performance metrics
  metrics_dt_dataframe <- get_dataframe("Decision Tree", metrics_dt)
  
  # Return a list containing the metrics dataframe and the raw metrics
  return(list(metrics_dt_dataframe = metrics_dt_dataframe, metrics_dt = metrics_dt, model_version = model_version))
}

XG_Boost <- function(train_data, test_data) {
  # library(data.table)
  # library(caret)
  # library(pROC)
  # library(xgboost)
  # library(parallel)
  
  # Convert train_data and test_data to data.table format for efficient processing
  setDT(train_data)
  setDT(test_data)
  
  # Calculate the original percentage of "yes" labels before any processing
  pct_yes <- mean(train_data$event == "yes")
  
  # Partition the train_data into training (80%) and validation (20%) sets
  set.seed(123) # Ensure reproducibility
  trainIndex <- createDataPartition(train_data$event, p = 0.8, list = FALSE)
  train_set <- train_data[trainIndex, ]
  validation_set <- train_data[-trainIndex, ]
  
  # Create model matrices for candidate 1 (no SMOTE)
  x_train <- model.matrix(event ~ . - 1, data = train_set)
  y_train <- as.numeric(train_set$event == "yes")
  x_validation <- model.matrix(event ~ . - 1, data = validation_set)
  y_validation <- as.numeric(validation_set$event == "yes")
  
  dtrain <- xgb.DMatrix(data = x_train, label = y_train)
  dvalidation <- xgb.DMatrix(data = x_validation, label = y_validation)
  
  scale_pos_weight1 <- sum(y_train == 0) / sum(y_train == 1)
  num_cores <- detectCores()
  
  params1 <- list(
    objective = "binary:logistic",
    eval_metric = "logloss",
    max_depth = 6,
    eta = 0.1,
    subsample = 0.8,
    colsample_bytree = 0.8,
    nthread = num_cores,
    scale_pos_weight = scale_pos_weight1
  )
  
  xgb_model_no_smote <- xgb.train(
    params = params1,
    data = dtrain,
    nrounds = 200,
    early_stopping_rounds = 10,
    watchlist = list(train = dtrain, validation = dvalidation),
    verbose = 0
  )
  
  val_preds_no_smote <- predict(xgb_model_no_smote, dvalidation)
  roc_val_no_smote <- roc(response = validation_set$event, predictor = val_preds_no_smote)
  auc_no_smote <- auc(roc_val_no_smote)
  optimal_threshold_no_smote <- coords(roc_val_no_smote, "best", ret = "threshold", best.method = "youden")
  optimal_threshold_no_smote <- as.numeric(optimal_threshold_no_smote)[1]
  
  if (pct_yes < 0.15) {
    # Candidate 2: With SMOTE
    train_set_smote <- smote_data(train_set)
    x_train_smote <- model.matrix(event ~ . - 1, data = train_set_smote)
    y_train_smote <- as.numeric(train_set_smote$event == "yes")
    dtrain_smote <- xgb.DMatrix(data = x_train_smote, label = y_train_smote)
    
    scale_pos_weight2 <- sum(y_train_smote == 0) / sum(y_train_smote == 1)
    
    params2 <- list(
      objective = "binary:logistic",
      eval_metric = "logloss",
      max_depth = 6,
      eta = 0.1,
      subsample = 0.8,
      colsample_bytree = 0.8,
      nthread = num_cores,
      scale_pos_weight = scale_pos_weight2
    )
    
    xgb_model_smote <- xgb.train(
      params = params2,
      data = dtrain_smote,
      nrounds = 200,
      early_stopping_rounds = 10,
      watchlist = list(train = dtrain_smote, validation = dvalidation),
      verbose = 0
    )
    
    val_preds_smote <- predict(xgb_model_smote, dvalidation)
    roc_val_smote <- roc(response = validation_set$event, predictor = val_preds_smote)
    auc_smote <- auc(roc_val_smote)
    optimal_threshold_smote <- coords(roc_val_smote, "best", ret = "threshold", best.method = "youden")
    optimal_threshold_smote <- as.numeric(optimal_threshold_smote)[1]
    
    if (auc_smote > auc_no_smote) {
      final_model <- xgb_model_smote
      final_threshold <- optimal_threshold_smote
      model_version <- "smote"
    }
    else {
      final_model <- xgb_model_no_smote
      final_threshold <- optimal_threshold_no_smote
      model_version <- "no smote"
    }
  }
  else {
    final_model <- xgb_model_no_smote
    final_threshold <- optimal_threshold_no_smote
    model_version <- "no smote"
  }
  
  x_test <- model.matrix(event ~ . - 1, data = test_data)
  dtest <- xgb.DMatrix(data = x_test)
  
  test_preds <- predict(final_model, dtest)
  binary_prediction_xgb <- ifelse(test_preds > final_threshold, "yes", "no")
  binary_prediction_xgb <- factor(binary_prediction_xgb, levels = c("yes", "no"))
  test_data$event <- factor(test_data$event, levels = c("yes", "no"))
  
  roc_curve <- roc(response = test_data$event, predictor = test_preds)
  auc_score <- auc(roc_curve)
  
  plot(roc_curve, main = paste("ROC Curve (AUC =", round(auc_score, 4), ")"))
  
  confusion_matrix_xgb <- table(
    factor(binary_prediction_xgb, levels = c("yes", "no")),
    factor(test_data$event, levels = c("yes", "no"))
  )
  
  if (!all(c("yes", "no") %in% rownames(confusion_matrix_xgb))) {
    confusion_matrix_xgb <- matrix(0, nrow = 2, ncol = 2, dimnames = list(c("yes", "no"), c("yes", "no")))
  }
  
  metrics_xgb <- calculate_model_metrics(confusion_matrix_xgb, test_preds, "XGBoost", auc_score)
  metrics_xgb_dataframe <- get_dataframe("XGBoost", metrics_xgb)
  
  return(list(metrics_xgb_dataframe = metrics_xgb_dataframe, metrics_xgb = metrics_xgb, model_version = model_version))
}

elastic_net <- function(train_data, test_data) {
  # library(glmnet)
  # library(data.table)
  # library(pROC)
  # library(caret)
  
  # Convert train_data and test_data to data.table format for fast operations
  setDT(train_data)
  setDT(test_data)
  
  # Calculate the original percentage of "yes" labels before any transformation
  pct_yes <- mean(train_data$event == "yes")
  
  # Partition train_data into a training set (80%) and a validation set (20%)
  set.seed(123)
  trainIndex <- createDataPartition(train_data$event, p = 0.8, list = FALSE)
  train_set <- train_data[trainIndex, ]
  validation_set <- train_data[-trainIndex, ]
  
  # Candidate 1: Without SMOTE
  x_train <- model.matrix(event ~ . - 1, data = train_set)
  y_train <- train_set$event
  x_validation <- model.matrix(event ~ . - 1, data = validation_set)
  
  elastic_net_model <- glmnet(x_train, y_train, family = "binomial", alpha = 0.5)
  cv_model <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 0.5)
  best_lambda <- cv_model$lambda.min
  
  predict_probabilities_val <- predict(elastic_net_model, s = best_lambda, newx = x_validation, type = "response")
  roc_val <- roc(response = validation_set$event, predictor = as.vector(predict_probabilities_val))
  auc_val <- auc(roc_val)
  optimal_threshold <- coords(roc_val, "best", ret = "threshold", best.method = "youden")
  optimal_threshold <- as.numeric(optimal_threshold)[1]
  
  if (pct_yes < 0.15) {
    # Candidate 2: With SMOTE
    train_set_smote <- smote_data(train_set)
    x_train_smote <- model.matrix(event ~ . - 1, data = train_set_smote)
    y_train_smote <- train_set_smote$event
    
    elastic_net_model_smote <- glmnet(x_train_smote, y_train_smote, family = "binomial", alpha = 0.5)
    cv_model_smote <- cv.glmnet(x_train_smote, y_train_smote, family = "binomial", alpha = 0.5)
    best_lambda_smote <- cv_model_smote$lambda.min
    
    predict_probabilities_val_smote <- predict(elastic_net_model_smote, s = best_lambda_smote, newx = x_validation, type = "response")
    roc_val_smote <- roc(response = validation_set$event, predictor = as.vector(predict_probabilities_val_smote))
    auc_val_smote <- auc(roc_val_smote)
    optimal_threshold_smote <- coords(roc_val_smote, "best", ret = "threshold", best.method = "youden")
    optimal_threshold_smote <- as.numeric(optimal_threshold_smote)[1]
    
    if (auc_val_smote > auc_val) {
      final_model <- elastic_net_model_smote
      final_lambda <- best_lambda_smote
      final_threshold <- optimal_threshold_smote
      model_version <- "smote"
    }
    else {
      final_model <- elastic_net_model
      final_lambda <- best_lambda
      final_threshold <- optimal_threshold
      model_version <- "no smote"
    }
  }
  else {
    final_model <- elastic_net_model
    final_lambda <- best_lambda
    final_threshold <- optimal_threshold
    model_version <- "no smote"
  }
  
  # Convert the test dataset to matrix format
  x_test <- model.matrix(event ~ . - 1, data = test_data)
  
  # Predict probabilities on the test set using the selected model
  predict_probabilities_test <- predict(final_model, s = final_lambda, newx = x_test, type = "response")
  binary_prediction_test <- ifelse(predict_probabilities_test > final_threshold, "yes", "no")
  
  # Compute the ROC curve and AUC for the test set predictions
  roc_curve <- roc(response = test_data$event, predictor = as.vector(predict_probabilities_test))
  auc_score <- auc(roc_curve)
  
  # Plot the ROC curve with the AUC value displayed in the title
  plot(roc_curve, main = paste("ROC Curve (AUC =", round(auc_score, 4), ")"))
  
  # Create a confusion matrix for the test set predictions
  test_conf_matrix <- table(Predicted = binary_prediction_test, Actual = test_data$event)
  
  # Evaluate test set performance using custom metrics
  metrics_en <- calculate_model_metrics(test_conf_matrix, predict_probabilities_test, "Elastic Net", auc_score)
  metrics_en_dataframe <- get_dataframe("Elastic Net", metrics_en)
  
  return(list(metrics_en_dataframe = metrics_en_dataframe, metrics_en = metrics_en, model_version = model_version))
}