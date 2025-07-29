library(readr)
library(dplyr)
library(RColorBrewer)
library(tidyverse)
library(lubridate)
library(survival)
library(survminer)
library(ggpubr)
library(data.table)
library(dotenv)

select <- dplyr::select
rename <- dplyr::rename
mutate <- dplyr::mutate
group_by <- dplyr::group_by
load_dot_env(".env")


# Depending on the environment, load transactions and other helper data from the proper data directory
if(Sys.getenv("DEVMODE") == "TRUE"){
  stableCoins <- read_csv("~/data/IDEA_DeFi_Research/Data/Coin_Info/stablecoins.csv")
  transactions <- read_rds("~/data/IDEA_DeFi_Research/Data/Lending_Protocols/Aave/V2/Mainnet/Experimental/transactions_user_market_time.rds") 
  
  
} else{
  stableCoins <- read_csv("./Data/Other_Data/stablecoins.csv")
  transactions <- read_csv("./Data/Raw_Transaction_Data_Sample/transactionsSample.csv") 
  
}

# cutoffDate <- 1704085199 # The end of 2023
cutoffDate_Q3_2024 = 1727755200 # the end of Q3 2024

transactions <- transactions %>%
  filter(timestamp <= cutoffDate_Q3_2024) %>%
  mutate(coinType = case_when(reserve %in% stableCoins$symbol ~ "Stable",
                              TRUE ~ "Non-Stable"))

# Load the survival data creation function:
source("./DeFi_source/Data_Creation_Functions/createSurvData.R")
source("./DeFi_source/Data_Creation_Functions/createTrainTestSplit.R")

## Helper functions
not_all_na <- function(x) any(!is.na(x))
`%notin%` <- Negate(`%in%`)


# These will be the default settings for subjects unless otherwise specified:
subjects <- c("user", "reserve")

#####
# Create survival data for basic transaction types:
#####
# We are filtering out liquidations, as those events are slightly more complicated and need to be handled separately from the basic transaction types.
# We also filter out flashLoans, collaterals, and swaps. These transaction types either lack sufficient information to be properly correlated to other
# transactions, or they are just not interesting transactions for the survival experiments we want to run.
basicTransactions <- transactions %>%
  filter(type != "liquidation",
         type != "collateral",
         type != "flashLoan",
         type != "swap")

basicEventTypes <- basicTransactions %>%
  dplyr::select(type) %>%
  distinct()


dataPath = "/data/IDEA_DeFi_Research/Data/Survival_Data_F24/"

buffer_duration = 30 # We chose a buffer duration of 30 days because most of the RMSTs were around that long.
train_cutoff = 1656648000 # The training cutoff is 2022/07/01
test_cutoff = 1719806400 # 2024/11/07
featuresToDrop <- c("userAlias",
                    "onBehalfOfAlias",
                    "liquidatorAlias") # If there are specific features we don't want to keep in our final published data, we should add them here

for(indexEvent in basicEventTypes$type){
  # Each index event should have its own directory. We create the appropriate directory
  # here, in case it doesn't already exist:
  
  dir.create(paste0(dataPath, str_to_title(indexEvent), "/"))
  
  # For each index event, let's save the train and test features at the top level:
  X_train <- transactions %>%
    filter(type == indexEvent) %>%
    filter(timestamp <= train_cutoff - buffer_duration) %>%
    select(-any_of(featuresToDrop))
  
  X_test <- transactions %>%
    filter(type == indexEvent) %>%
    filter(timestamp > train_cutoff & timestamp <= test_cutoff-buffer_duration) %>%
    select(-any_of(featuresToDrop))
  
  X_train_size = as.numeric(object.size(X_train))
  X_train_chunks = ceiling(X_train_size / (50*1000*1000)) # partition into ~50Mb chunks
  X_train_chunk_size = ceiling(length(X_train[[1]]) / X_train_chunks)
  
  X_test_size = as.numeric(object.size(X_test))
  X_test_chunks = ceiling(X_test_size / (50*1000*1000))
  X_test_chunk_size = ceiling(length(X_test[[1]]) / X_test_chunks)
  
  dir.create(paste0(dataPath, str_to_title(indexEvent), "/X_train/"))
  dir.create(paste0(dataPath, str_to_title(indexEvent), "/X_test/"))
  
  for(i in 1:X_train_chunks){
    chunk <- X_train[((i-1)*X_train_chunk_size + 1):min(i*X_train_chunk_size, length(X_train[[1]])),]
    write_rds(chunk, paste0(dataPath, str_to_title(indexEvent), "/X_train/X_train_", i, ".rds"))
  }
  for(i in 1:X_test_chunks){
    chunk <- X_test[((i-1)*X_test_chunk_size + 1):min(i*X_test_chunk_size, length(X_test[[1]])),]
    write_rds(chunk, paste0(dataPath, str_to_title(indexEvent), "/X_test/X_test_", i, ".rds"))
  }
  
  for(outcomeEvent in basicEventTypes$type){

    if(indexEvent == outcomeEvent){
      next # We skip the case when the index and outcome events are the same because this causes issues with the rolling join used to create the survival data
    }
    
    # Each outcome event should have its own directory within the index event's folder:
    dir.create(paste0(dataPath, str_to_title(indexEvent), "/", str_to_title(outcomeEvent), "/"))
    build_survival_data(transactions,
                        indexEvent,
                        outcomeEvent,
                        subjects,
                        train_cutoff,
                        test_cutoff,
                        buffer_duration,
                        paste0(dataPath, str_to_title(indexEvent), "/", str_to_title(outcomeEvent), "/"))
  }
  
  # # Let's also create the outcome events of accountLiquidated and liquidationPerformed:
  outcomeEvent = "account liquidated"
  dir.create(paste0(dataPath, str_to_title(indexEvent), "/", str_to_title(outcomeEvent), "/"))

  liquidations <- transactions %>%
    filter(type == "liquidation") %>%
    mutate(type = "account liquidated") # Change the label of the transaction to be accountLiquidated

  transactionsWithAL <- basicTransactions %>%
    bind_rows(liquidations)

  build_survival_data(transactionsWithAL,
                      indexEvent,
                      outcomeEvent,
                      subjects = c("user"),
                      train_cutoff,
                      test_cutoff,
                      buffer_duration,
                      paste0(dataPath, str_to_title(indexEvent), "/", str_to_title(outcomeEvent), "/"))

  
}


