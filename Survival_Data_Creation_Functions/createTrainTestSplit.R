build_survival_data <- function(transaction_data, 
                                indexEvent, 
                                outcomeEvent, 
                                subjects, 
                                train_cutoff,
                                test_cutoff, 
                                buffer_duration,
                                dataPath) {
  # load necessary libraries
  library(dplyr) # for data manipulation functions like filter(), bind_rows()
  library(readr) # for reading and writing .rds files with read_rds() and saveRDS()
  
  # load the external r script to access the createSurvData function
  source("~/DAR-DeFi-LTM-F24/DeFi_source/Data_Creation_Functions/createSurvData.R")
  
  
  # generate train_transaction_data by filtering for events within the training period
  # filter for the indexEvent within the training period
  train_indexEvent <- transaction_data %>% filter(type == indexEvent) %>%
    filter(timestamp <= (train_cutoff - buffer_duration))
  
  # filter rows where train_cutoff >= timestamp for outcomeEvent in training period
  train_outcomeEvent <- transaction_data %>% filter(type == outcomeEvent) %>%
    filter(train_cutoff >= timestamp)
  
  # combine index and outcome events for the training set
  train_transaction_data <- bind_rows(train_indexEvent, train_outcomeEvent)
  
  # generate test_transaction_data by filtering for events within the testing period
  # filter for the indexEvent within the testing period
  test_indexEvent <- transaction_data %>% filter(type == indexEvent) %>%
    filter(timestamp <= (test_cutoff - buffer_duration) & (timestamp > train_cutoff))
  
  # filter rows where (test_cutoff >= timestamp) & (timestamp > train_cutoff) for
  # outcomeEvent in testing period
  test_outcomeEvent <- transaction_data %>% filter(type == outcomeEvent) %>%
    filter((test_cutoff >= timestamp) & (timestamp > train_cutoff))
  
  # combine index and outcome events for the testing set
  test_transaction_data <- bind_rows(test_indexEvent, test_outcomeEvent)
  
  # set train and test data as the main data sources for survival analysis
  train_data <- train_transaction_data
  test_data <- test_transaction_data
  
  # define subjects involved in the survival analysis, such as "user" and "reserve"
  subjects <- c("user", "reserve")
  
  # set observation period as default, can adjust if needed
  observationPeriod <- c(0, -1)
  
  # define covariates to be included for index events and leave outcome covariates empty
  indexCovariates <- c("id")
  outcomeCovariates <- c()
  
  # create the training survival data using the createSurvData function with specified parameters
  train_survival_data <- createSurvData(
    indexEventSet = indexEvent,
    outcomeEventSet = outcomeEvent,
    data = train_data,
    subjects = subjects,
    observationPeriod = observationPeriod,
    indexCovariates = indexCovariates,
    outcomeCovariates = outcomeCovariates
  ) %>%
    select(timeDiff, status, id, `Index Event`, `Outcome Event`)
  
  # create the testing survival data using the createSurvData function with specified parameters
  test_survival_data <- createSurvData(
    indexEventSet = indexEvent,
    outcomeEventSet = outcomeEvent,
    data = test_data,
    subjects = subjects,
    observationPeriod = observationPeriod,
    indexCovariates = indexCovariates,
    outcomeCovariates = outcomeCovariates
  ) %>%
    select(timeDiff, status, id, `Index Event`, `Outcome Event`)
  
  # set the paths to store the generated train and test survival data
  file_path_y_train <- paste0(dataPath, "/y_train.rds")
  file_path_y_test <- paste0(dataPath, "/y_test.rds")
  
  # save the created training and testing survival data to the specified file paths
  saveRDS(train_survival_data, file = file_path_y_train)
  saveRDS(test_survival_data, file = file_path_y_test)
}