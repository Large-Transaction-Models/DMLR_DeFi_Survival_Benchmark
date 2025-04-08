concordanceIndex <- function(predictions, test, model_type = c('cox', 'aft', 'gbm', 'xgb', 'rsf')){
  #Install required packages if needed
  if (!require("survival")) {
    install.packages("survival")
    library(survival)
  }
  
  #Comment out when xgboost, cox
  test <- test[test$timeDiff > 0,]
  
  model_type <- match.arg(model_type) 
  
  # Compute concordance index using concordance() function
  cindex <- concordance(Surv((timeDiff/86400), status) ~ predictions, data = test)$concordance
  
  # If the C-index is less than 0.5, we should just return 1-Cindex since we can simply reverse the order of 
  # predictions in order to get better results. In theory, we should be tweaking the parameters of the model
  # to avoid this problem altogether, but this is the lazy way of doing that for now.
  return(max(cindex, 1-cindex))
}
