# Author: Aaron Green
library(fastDummies)
library(dplyr)
library(conflicted)

conflict_prefer("filter", "dplyr")
conflict_prefer("summarize", "dplyr")
conflict_prefer("select", "dplyr")



# This function is tailored specifically for our DeFi data and does not necessarily use best coding practices.
# It should be used in the following way:
#   After loading a train and test set using dataLoader.R, pass the train data into this function
#   and make sure to parse the output for all three results, which will include the one-hot-encoded
#   training data and lists which represent the "smart-one-hot-encoded" categories for reserve and userReserveMode.
#
#   Next, pass in the test data along with the lists for reserves and userReserveMode in order to 
#   one-hot-encode the testing data with the same categories as the training data. This is important!!!
#   
smartOneHotEncode <- function(df, topReserveTypes = list(), topUserReserveModes = list()){
  # We should convert categorical features into one-hot encoded features, and make sure that we don't have too many unnecessary features as such.
  
  factor_cols <- names(df)[sapply(df, is.factor)]
  
  
  #   reserve: For this, let's just make categories for the top ten reserve types, and just put the rest in "other"
  if(length(topReserveTypes) == 0){
    topReserveTypes <- df %>%
      select(reserve) %>%
      group_by(reserve) %>%
      summarize(count = n()) %>%
      ungroup() %>%
      arrange(-count) %>%
      head(10) %>%
      select(reserve)
    
    topReserveTypes <- as.list(as.character(topReserveTypes$reserve))  
    
  }
  
  df <- df %>%
    mutate(reserve = case_when(reserve %in% topReserveTypes ~ reserve,
                               TRUE ~ "Other"))
  
  #   userReserveMode: We will handle this the same as reserve.
  if(length(topUserReserveModes) == 0){
    topUserReserveModes <- df %>%
      select(userReserveMode) %>%
      group_by(userReserveMode) %>%
      summarize(count = n()) %>%
      ungroup() %>%
      arrange(-count) %>%
      head(10) %>%
      select(userReserveMode) 
    
    topUserReserveModes <- as.list(as.character(topUserReserveModes$userReserveMode))
  }
  
  
  df <- df %>%
    mutate(userReserveMode = case_when(userReserveMode %in% topUserReserveModes ~ userReserveMode,
                                       TRUE ~ "Other"))
  
  # Now we use the fastDummies package to quickly transform reserve and userReserveMode into one-hot-encodings:
  
  df_encoded <- dummy_cols(
    df,
    select_columns = c("reserve", "userReserveMode", factor_cols),  # Which column(s) to encode
    remove_selected_columns = TRUE,  # Drop the original columns
    remove_first_dummy = TRUE        # Often set to TRUE to avoid dummy trap
  )
  
  return(list(df_encoded, topReserveTypes, topUserReserveModes))
}


preprocess <- function(train, test,
                       useScaling = TRUE,
                       useOneHotEncoding = TRUE,
                       usePCA = TRUE, pcaExplainedVar = 0.9, 
                       classificationTask = FALSE, classificationCutoff = -1){
  
  
  # Let's save off the target columns up front so we can drop them before scaling:
  trainTargets <- train %>%
    select(timeDiff, status)
  testTargets <- test %>%
    select(timeDiff, status)
  # Let's drop some of the columns that we know we can't use:
  cols_to_drop = c("timeDiff", "status",
                   "id", "Index Event", "Outcome Event",
                   "type", "pool",
                   "user", "timestamp")
  
  train <- train %>%
    select(-any_of(cols_to_drop)) %>%
    mutate(across(where(is.character), as.factor))
  test <- test %>%
    select(-any_of(cols_to_drop)) %>%
    mutate(across(where(is.character), as.factor))
  
  
  
  trainDataCategoricalCols <- train %>%
    select(where(is.factor))
  testDataCategoricalCols <- test %>%
    select(where(is.factor))
  train <- train %>%
    select(-where(is.factor))
  test <- test %>%
    select(-where(is.factor))
  
  if(useScaling == TRUE){
    # Let's scale the data:
    train <- scale(as.matrix(train))
    
    test <- data.frame(scale(as.matrix(test), center=attr(train, "scaled:center"), scal=attr(train, "scaled:scale")))
    
    train <- data.frame(train) 
    test <- data.frame(test)
    
    train <- train[ , !sapply(train, function(x) all(is.na(x)))]
    
    common_cols <- intersect(colnames(train), colnames(train))
    test <- test %>%
      select(all_of(common_cols))
    
  }
  
  if(useOneHotEncoding == TRUE){
    # One-hot encode the categorical data if requested:
    trainOutput <- smartOneHotEncode(trainDataCategoricalCols)
    trainDataCategoricalCols <- trainOutput[[1]]
    topReserveTypes <- trainOutput[[2]]
    topUserReserveModes <- trainOutput[[3]]
    testDataCategoricalCols <- smartOneHotEncode(testDataCategoricalCols, 
                                                 topReserveTypes = topReserveTypes, 
                                                 topUserReserveModes = topUserReserveModes)[[1]]
  }
  
  # Put categorical columns back in the data:
  train <- train %>%
    bind_cols(trainDataCategoricalCols)
  
  test <- test %>%
    bind_cols(testDataCategoricalCols)
  
  
  
  
  if(usePCA == TRUE){
    # Now that we have scaled, encoded data, let's run PCA to help eliminate collinearity of features:
    pca_result <- prcomp(train, center = FALSE, scale. = FALSE)
    
    
    # Variances of each PC are the squared standard deviations:
    pc_variances <- pca_result$sdev^2  
    
    # Proportion of total variance explained by each PC:
    prop_variance <- pc_variances / sum(pc_variances)
    
    # Cumulative variance explained:
    cumvar <- cumsum(prop_variance)
    
    # Find the smallest number of PCs explaining at least 90% variance:
    num_pcs <- which(cumvar >= pcaExplainedVar)[1]
    
    # Keep only the PCs that explain ≥ 90% of variance
    train <- as.data.frame(pca_result$x[, 1:num_pcs])
    
    # 'scores_90pct' is now a matrix with the same number of rows as df,
    # but fewer columns (one column per principal component).
    test <- data.frame(predict(pca_result, newdata = test))[, 1:num_pcs]
  }
  
  
  # Put the targets back in the data:
  final_train_data <- train %>%
    bind_cols(trainTargets)
  
  final_test_data <- test %>%
    bind_cols(testTargets)
  
  
  
  if(classificationTask){
    # filter out invalid records where `timeDiff` is <= 0 early
    final_train_data <- final_train_data %>% filter(timeDiff > 0)
    
    # filter out records based on the `set_timeDiff` threshold and `status`
    final_train_data <- final_train_data %>% filter(!(timeDiff / 86400 <= classificationCutoff & status == 0))
    
    # create a new binary column `event` based on `timeDiff`
    final_train_data <- final_train_data %>%
      mutate(event = case_when(
        timeDiff / 86400 <= classificationCutoff ~ "yes",
        timeDiff / 86400 > classificationCutoff ~ "no"
      )) %>%
      select(-timeDiff, -status)
    
    # filter out invalid records where `timeDiff` is <= 0 early
    final_test_data <- final_test_data %>% filter(timeDiff > 0)
    
    # filter out records based on the `set_timeDiff` threshold and `status`
    final_test_data <- final_test_data %>% filter(!(timeDiff / 86400 <= classificationCutoff & status == 0))
    
    # create a new binary column `event` based on `timeDiff`
    final_test_data <- final_test_data %>%
      mutate(event = case_when(
        timeDiff / 86400 <= classificationCutoff ~ "yes",
        timeDiff / 86400 > classificationCutoff ~ "no"
      )) %>%
      select(-timeDiff, -status)
  }
  
  return(list(final_train_data, final_test_data))
}