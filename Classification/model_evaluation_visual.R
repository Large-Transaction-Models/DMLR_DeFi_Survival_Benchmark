library(dplyr)
library(data.table)
library(ggplot2)
library(scales)
library(reshape2)
library(caret)

calculate_model_metrics <- function(confusion_matrix, binary_predictions, model_name, auc_score) {
  # Extract True Negatives, False Positives, False Negatives, and True Positives from the confusion matrix
  TN <- confusion_matrix[1, 1] # True Negatives
  FP <- confusion_matrix[1, 2] # False Positives
  FN <- confusion_matrix[2, 1] # False Negatives
  TP <- confusion_matrix[2, 2] # True Positives
  
  # Calculate Specificity (True Negative Rate)
  specificity <- TN / (TN + FP)
  
  # Calculate Sensitivity (Recall, True Positive Rate)
  sensitivity <- TP / (TP + FN)
  
  # Calculate Balanced Accuracy as the average of Sensitivity and Specificity
  balanced_accuracy <- (specificity + sensitivity) / 2
  
  # Calculate Precision
  precision <- TP / (TP + FP)
  
  # Calculate F1 Score as the harmonic mean of Precision and Sensitivity
  f1_score <- 2 * (precision * sensitivity) / (precision + sensitivity)
  
  if (is.nan(balanced_accuracy)) balanced_accuracy <- 0.50
  if (is.nan(f1_score)) f1_score <- 0.50
  if (is.nan(auc_score)) auc_score <- 0.50
  
  # Print all performance metrics with labels
  print(paste(model_name, "model prediction accuracy:"))
  cat("Balanced Accuracy:", sprintf("%.3f", balanced_accuracy), "\n")
  cat("F1 Score:", sprintf("%.3f", f1_score), "\n")
  cat("AUC Score:", sprintf("%.3f", auc_score), "\n")
  
  # Return all computed metrics in a list
  return(list(
    balanced_accuracy = balanced_accuracy, 
    f1_score = f1_score,
    auc_score = auc_score
  ))
}

get_dataframe <- function(model_name, metrics) {
  metrics_dataframe <- data.frame(
    Model = model_name, 
    # balanced_accuracy = sprintf("%.2f%%", metrics$balanced_accuracy * 100),
    auc_score = sprintf("%.3f", metrics$auc_score),
    f1_score = sprintf("%.3f", metrics$f1_score)
  )
  return (metrics_dataframe)
}

combine_classification_results <- function(accuracy_dataframe_list, data_combination) {
  # Remove NULL dataframes to ensure the list contains only valid dataframes
  accuracy_dataframe_list <- accuracy_dataframe_list[!sapply(accuracy_dataframe_list, is.null)]
  
  # Add the `Data_Combination` column to each dataframe in the list
  accuracy_dataframe_list <- lapply(accuracy_dataframe_list, function(df) {
    df$Data_Combination <- data_combination
    return(df)
  })
  
  # Merge all dataframes into a single dataframe
  combined_dataframe <- do.call(rbind, accuracy_dataframe_list)
  
  return(combined_dataframe)
}

get_percentage <- function(survivalDataForClassification, indexEvent, outcomeEvent) {
  # indexEvent and outcomeEvent is a string type
  pctPerEvent <- survivalDataForClassification %>%
    group_by(event) %>%
    dplyr::summarize(numPerEvent = n()) %>%
    mutate(total = sum(numPerEvent)) %>%
    mutate(percentage = numPerEvent / total) %>%
    dplyr::select(event, percentage)
  # create a bar plot for event percentages
  # stat = "identity": percentages used directly to draw the bar chart
  print(ggplot(pctPerEvent, aes(x = event, y = percentage, fill = event)) +
          geom_bar(stat = "identity") +
          scale_y_continuous(labels = scales::percent_format()) +  # show y-axis in percentage
          labs(title = "Percentage of Events: 'Yes' event vs 'No' event",
               x = paste(indexEvent, "and", outcomeEvent),
               y = "Percentage") +
          geom_text(aes(label = scales::percent(percentage)), 
                    vjust = -0.5, size = 3.5) +  # show percentages on top of bars
          theme_minimal())
}

accuracy_comparison_plot <- function(metrics_list) {
  # initialize an empty data frame to store the metrics for all models
  accuracy_table <- data.frame()
  
  # loop over each element in metrics_list (each element is a list containing metrics and model name)
  for (metrics in metrics_list) {
    # Extract metrics and model name from each "tuple"
    model_metrics <- metrics[[1]]
    model_name <- metrics[[2]]
    
    # create a temporary dataframe for this model
    temp_df <- data.frame(
      Model = model_name, 
      # balanced_accuracy = model_metrics$balanced_accuracy,
      auc_score = model_metrics$auc_score,
      f1_score = model_metrics$f1_score
    )
    
    # append the temporary dataframe to the main accuracy_table
    accuracy_table <- rbind(accuracy_table, temp_df)
  }
  
  # melt the dataframe into long format for plotting
  accuracy_results_melted <- reshape2::melt(accuracy_table, id.vars = "Model")
  
  # generate the plot with faceted bars
  ggplot(accuracy_results_melted, aes(x = Model, y = value, fill = Model)) +
    geom_bar(stat = "identity", position = "dodge") +
    facet_wrap(~ variable, scales = "free_y") +  # Facet by each metric
    labs(title = "Comparison of Accuracy Metrics Across Models",
         x = "Model",
         y = "Value") +
    # add percentage labels on top of each bar
    geom_text(aes(label = scales::percent(value, accuracy = 0.1)),
              position = position_dodge(width = 0.9),
              vjust = 0.5, size = 2.0) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

specific_accuracy_table <- function(event_pair, accuracy_type, metrics_list) {
  # initialize the result list
  results <- list()
  
  # add accuracy_type as the left front corner and event_pair is the first column of the data
  results[[accuracy_type]] <- event_pair
  
  # traverse metrics_list and extract the specified accuracy type for each model
  for (metric_item in metrics_list) {
    # get accuracy data
    accuracy_data <- metric_item[[1]]
    # get the model name
    model_name <- metric_item[[2]]
    if (accuracy_type == "balanced_accuracy") {
      results[[model_name]] <- round(accuracy_data$balanced_accuracy, 3)
    }
    else if (accuracy_type == "auc_score") {
      results[[model_name]] <- round(accuracy_data$auc_score, 3)
    }
    else if (accuracy_type == "f1_score") {
      results[[model_name]] <- round(accuracy_data$f1_score, 3)
    }
    else {
      # An error message is displayed if the specified accuracy_type does not exist.
      stop(paste("Invalid accuracy type:", accuracy_type))
    }
  }
  
  # convert the result to a DataFrame and set row.names = NULL
  df <- as.data.frame(results, row.names = NULL)
  return(df)
}

specific_model_version_table <- function(event_pair, model_version_list) {
  # initialize the results list without punctuation
  results <- list()
  
  # add model_version label in the first column using event_pair
  results[["model_version"]] <- event_pair
  
  # traverse the model_version_list where each element contains the metrics and model name
  for (metric_item in model_version_list) {
    # extract model metrics and model name from the list element
    model_version <- metric_item[[1]]
    model_name <- metric_item[[2]]
    
    # assign the model_version from model metrics
    results[[model_name]] <- model_version
  }
  
  # convert the results list to a data frame and return it
  df <- as.data.frame(results, row.names = NULL)
  return(df)
}

combine_accuracy_dataframes <- function(df_list) {
  # check if the input is a list
  if (!is.list(df_list)) {
    stop("Input must be a list of data.frames.")
  }
  
  # Use do.call and rbind to combine all data.frames in a list.
  combined_df <- do.call(rbind, df_list)
  
  # returns the merged data.frame
  return(combined_df)
}