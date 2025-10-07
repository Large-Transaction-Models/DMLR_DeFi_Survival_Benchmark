cox_regression <- function(train, test, features = NULL){
  
  #Install required packages if needed
  if (!require("survival")) {
    install.packages("survival")
    library(survival)
  }
  if (!require("tidyverse")) {
    install.packages("tidyverse")
    library(tidyverse)
  }
  
  
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
  print(nrow(test.X))
  
  # Train Cox Model
  coxModel = coxph(Surv(timeDiff/86400, status)~ ., 
                   data = train, x = TRUE)
  
  
  # Predict using the trained Cox model and the data found in test.X
  cox_predictions = predict(coxModel, newdata = test, type="lp")
  
  l <- list(cox_predictions, coxModel)
  
  # Return the predictions made
  return(l)
}