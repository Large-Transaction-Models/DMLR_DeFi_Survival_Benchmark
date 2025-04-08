aft_regression <- function(train, test, features = NULL){
  
  if (!require("survival")) {
    install.packages("survival")
    library(survival)
  }
  if (!require("tidyverse")) {
    install.packages("tidyverse")
    library(tidyverse)
  }
  if (!require("purrr")) {
    install.packages("purrr")
    library(purrr)
  }
  
  # Confirm both train and test do not have rows with timeDiff = 0
  train <- train[train$timeDiff > 0,]
  test <- test[test$timeDiff > 0,]
  
  
  
  # Match levels in factor columns from testing to those from the training set.
  excluded_columns <- c("status")
  
  test <- test %>%
    mutate(across(
      .cols = where(is.factor) & !all_of(excluded_columns),
      .fns = ~ factor(., levels = levels(train[[cur_column()]]))
    ))
  
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
  
  # Fit the Accelerated failure time model
  aft_model <- survreg(Surv((timeDiff/86400), status) ~ ., 
                       data = train, dist = "weibull")
  
  # Predict aft on test data
  aft_prediction <- predict(aft_model,newdata = test.X)/86400
  
  l <- list(aft_prediction, aft_model)
  
  return(l)
}