xgboost_regression <- function(trainData, testData) {
  
  # Install required packages if needed
  if (!require("xgboost")) {
    install.packages("xgboost")
    library(xgboost)
  }
  if (!require("Matrix")) {
    install.packages("Matrix")
    library(Matrix)
  }
  
  # Ensure trainData and testData are numeric where needed
  trainData <- trainData %>%
    mutate(across(where(is.character), as.numeric)) %>%
    mutate(across(where(is.factor), as.numeric))
  
  testData <- testData %>%
    mutate(across(where(is.character), as.numeric)) %>%
    mutate(across(where(is.factor), as.numeric))
  
  # Define labels for AFT model
  labels_lower <- trainData$timeDiff / 86400
  labels_upper <- ifelse(trainData$status == 1, trainData$timeDiff / 86400, Inf)
  
  # Convert categorical variables in trainData into one-hot encoded features
  features_train <- sparse.model.matrix(
    ~ . - timeDiff - status,  # Exclude timeDiff and status from features
    data = trainData
  )
  
  # Create DMatrix for XGBoost for training
  dtrain <- xgb.DMatrix(data = features_train)
  setinfo(dtrain, "label_lower_bound", labels_lower)
  setinfo(dtrain, "label_upper_bound", labels_upper)
  
  # Set up parameters for the AFT model
  params <- list(
    objective = "survival:aft",
    eval_metric = "aft-nloglik",
    aft_loss_distribution = "logistic",  
    aft_loss_distribution_scale = 1.0,
    max_depth = 3,
    eta = 0.01,        # Learning rate
    verbosity = 1
  )
  
  # Train the XGBoost model
  bst <- xgb.train(
    params = params,
    data = dtrain,
    nrounds = 1000,    # Number of boosting rounds
    verbose = 0,
    watchlist = list(train = dtrain),
    early_stopping_rounds = 50  # To prevent overfitting
  )
  
  # Prepare the test data for predictions
  features_test <- sparse.model.matrix(
    ~ . - timeDiff - status,  # Exclude timeDiff and status from features
    data = testData
  )
  
  # Create DMatrix for the test data
  dtest <- xgb.DMatrix(data = features_test)
  
  # Make predictions on the test data
  predictions <- predict(bst, newdata = dtest)
  
  # Return the predictions and the model
  return(list(predictions = predictions, model = bst, features_test = features_test))
}
