library(readr)
library(stringr)
library(dplyr)
library(tidyverse)
library(tidyr)
library(ROSE)

get_train_test_data <- function(indexEvent, outcomeEvent) {
  source("~/KDD_DeFi_Survival_Dataset_And_Benchmark/dataLoader.R")
}

smote_data <- function(train_data, target_var = "event", seed = 123) {
  # library(ROSE)
  # check if the input data contains the target variable
  if (!target_var %in% colnames(train_data)) {
    stop(paste("Target variable", target_var, "not found in the dataset"))
  }
  
  # set the random seed (if provided)
  if (!is.null(seed)) {
    set.seed(seed)
  }
  
  # dynamic formula creation to adapt to different target variables
  formula <- as.formula(paste(target_var, "~ ."))
  
  # applying ROSE Balance Data
  train_data_balanced <- ROSE(formula, data = train_data, seed = seed)$data
  
  # return the balanced dataset
  return(train_data_balanced)
}