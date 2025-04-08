gbm_regression <- function(train, test, features = NULL){
  
  # Install required packages
  if (!require("survival")) {
    install.packages("survival")
    library(survival)
  }
  if (!require("tidyverse")) {
    install.packages("tidyverse")
    library(tidyverse)
  }
  if (!require("gbm")) {
    install.packages("gbm")
    library(gbm)
  }
  
  # Confirm there are no rows with timeDiff = 0 in the train and test data frames.
  train <- train[train$timeDiff > 0,]
  test <- test[test$timeDiff > 0,]
  
  # Convert logical features to factors in both train and test data frames
  train <- train %>%
    mutate(across(where(is.logical), as.factor))
  
  test <- test %>%
    mutate(across(where(is.logical), as.factor))
  
  
  if (!is.null(features)) {
    train <- train %>% select(any_of(c(features, "timeDiff", "status")))
    test <- test %>% select(any_of(c(features, "timeDiff", "status")))
    cat("Feature selection applied. \n")
  } else {
    cat("No feature selection applied. Using all features.\n")
  }
  
  # Create data frame without "status" and "timeDiff" features, for prediction
  test <- as.data.frame(test)
  test.X <- test[ , !(names(test) %in% c("status", "timeDiff"))]
  test.X <- as.data.frame(test.X)
  
  
  #Fit Gradient Boosting Survival Model
  gbm_model <- gbm(Surv(timeDiff/86400, status) ~ ., 
                   data = train, 
                   distribution = "coxph",
                   interaction.depth = 3,
                   n.trees = 300,
                   shrinkage = 0.01)
  
  print("feature importance")
  
  #feature_importance <- summary(gbm_model, n.trees = gbm.perf(gbm_model, method = "cv"))
  
  print("predictions")
  
  #Predict on test set
  gbm_prediction <- predict(gbm_model, newdata = test.X)
  
  l <- list(gbm_prediction, gbm_model)
  
  return(l)
}