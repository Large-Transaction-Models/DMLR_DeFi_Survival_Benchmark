# Author: Aaron Green
library(readr)
library(stringr)
library(dplyr)


## Helper functions
not_all_na <- function(x) any(!is.na(x))
`%notin%` <- Negate(`%in%`)


loadSurvivalDataset <- function(indexEvent, outcomeEvent, 
                                dataPath = "~/KDD_DeFi_Survival_Dataset_And_Benchmark/Data/Survival_Data/", 
                                X_path = "/X_train/",
                                y_path = "y_train.rds"){
  
  
  X_files <- list.files(paste0(dataPath, str_to_title(indexEvent), X_path), pattern = NULL, all.files = FALSE, full.names = TRUE)
  X = data.frame()
  
  for(file in X_files){
    X <- X %>%
      bind_rows(read_rds(file)) %>%
      select(where(not_all_na)) %>%
      select(-starts_with("exo")) %>%
      filter(!is.na(id))
  }
  
  y <- read_rds(paste0(dataPath, str_to_title(indexEvent), "/", str_to_title(outcomeEvent), "/", y_path)) %>%
    filter(!is.na(id))
  
  
  
  return(inner_join(y, X, by = "id"))
  
}

train = loadSurvivalDataset(indexEvent, outcomeEvent, X_path = "/X_train/", y_path = "y_train.rds")
test = loadSurvivalDataset(indexEvent, outcomeEvent, X_path = "/X_test/", y_path = "y_test.rds")


# Find the shared columns
shared_columns <- intersect(names(train), names(test))

# Subset each dataframe
train <- train %>% select(all_of(shared_columns))
test <- test %>% select(all_of(shared_columns))