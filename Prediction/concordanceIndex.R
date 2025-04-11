concordanceIndex <- function(predictions, test, model_type = c('cox', 'aft', 'gbm', 'xgb', 'rsf')) {
  # Install required packages if needed
  if (!require("survival")) {
    install.packages("survival")
    library(survival)
  }
  
  model_type <- match.arg(model_type)
  
  # Only filter 'test' when model_type is not xgb or cox to match the lengths
  if (!(model_type %in% c("xgb"))) {
    test <- test[test$timeDiff > 0,]
  }
  
  # Compute concordance index using the concordance() function from the survival package
  cindex <- concordance(Surv((timeDiff/86400), status) ~ predictions, data = test)$concordance
  
  
  return(cindex)
}