random_survival_forest <- function(train, test, features = NULL) {
  # Install required packages
  if (!require("survival")) {
    install.packages("survival")
    library(survival)
  }
  if (!require("tidyverse")) {
    install.packages("tidyverse")
    library(tidyverse)
  }
  if (!require("randomForestSRC")) {
    install.packages("randomForestSRC")
    library(randomForestSRC)
  }
 
  
  rsf_model <- rfsrc.fast(Surv((timeDiff/86400), status) ~ ., 
                     data = train, forest=TRUE)
  
  pred <- predict(rsf_model, newdata = test, importance = "none")
  predictions <- pred$predicted
  
  
  # Return the predictions and the model
  return(list(predictions = predictions, model = rsf_model))
  
  
}
