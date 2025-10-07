concordanceIndex <- function(predictions, test, hazard=FALSE) {
  # Install required packages if needed
  if (!require("survival")) {
    install.packages("survival")
    library(survival)
  }
  
  # Compute concordance index using the concordance() function from the survival package
  cindex <- concordance(Surv((timeDiff/86400), status) ~ predictions, data = test, reverse=hazard)$concordance
  
  
  return(cindex)
}