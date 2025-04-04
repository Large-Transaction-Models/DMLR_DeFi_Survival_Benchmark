get_classification_cutoff <- function(indexEvent, outcomeEvent){
  library(stringr)
  # Create the dataframe
  data <- data.frame(
    IndexEvent = c("Borrow", "Borrow", "Borrow", "Borrow", "Borrow", "Borrow", 
                   "Deposit", "Deposit", "Deposit", "Deposit", "Deposit",
                   "Repay", "Repay", "Repay", "Repay", "Repay",
                   "Withdraw", "Withdraw", "Withdraw", "Withdraw", "Withdraw"),
    OutcomeEvent = c("Account Liquidated", "Deposit", "Full Repay", "Liquidation Performed", "Repay", "Withdraw", 
                     "Account Liquidated", "Borrow", "Liquidation Performed", "Repay", "Withdraw",
                     "Account Liquidated", "Borrow", "Deposit", "Liquidation Performed", "Withdraw",
                     "Account Liquidated", "Borrow", "Deposit", "Liquidation Performed", "Repay"),
    ConvergedTau_1 = c(91, 99, 95, 101, 69, 99, 
                       90, 100, 101, 99, 93, 
                       93, 95, 100, 101, 100, 
                       94, 101, 99, 101, 100),
    ConvergedRMST_1 = c(69.36, 90.98, 74.12, 100.88, 28.71, 92.59, 
                        68.64, 91.58, 100.95, 89.70, 64.84, 
                        72.56, 63.23, 93.72, 100.86, 93.93,
                        75.10, 96.01, 82.13, 100.88, 93.82),
    ConvergedTau_5 = c(20, 21, 20, 21, 17, 21, 
                       20, 21, 21, 21, 20, 
                       20, 20, 21, 21, 21,
                       20, 21, 21, 21, 21),
    ConvergedRMST_5 = c(17.43, 19.88, 17.15, 20.99, 10.24, 20.21, 
                        17.52, 19.73, 20.99, 19.69, 15.67, 
                        17.43, 14.99, 20.13, 20.99, 20.15,
                        17.72, 20.26, 18.14, 20.98, 20.04),
    ConvergedTau_10 = c(11, 11, 11, 11, 10, 11, 
                        11, 11, 11, 11, 11, 
                        11, 11, 11, 11, 11,
                        11, 11, 11, 11, 11),
    ConvergedRMST_10 = c(9.94, 10.51, 9.72, 11.00, 6.66, 10.68, 
                         10.00, 10.43, 11.00, 10.44, 8.95, 
                         9.90, 8.64, 10.62, 11.00, 10.62,
                         10.05, 10.68, 9.68, 10.99, 10.57)
  )
  
  
  return(data %>% filter(IndexEvent == str_to_title(indexEvent), OutcomeEvent == str_to_title(outcomeEvent)) %>% pull(ConvergedRMST_5))
}