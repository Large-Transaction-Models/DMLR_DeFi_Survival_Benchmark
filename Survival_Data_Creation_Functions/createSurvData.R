
createSurvData <- function(indexEventSet, 
                           outcomeEventSet, 
                           data, 
                           subjects, 
                           observationPeriod = c(0, -1), 
                           indexCovariates = c(), 
                           outcomeCovariates = c()){
  require(data.table)
  survivalData <- NULL
  
  timeStart = as.double(max(observationPeriod[1], min(data$timestamp)))
  
  if(observationPeriod[2] == -1){
    # For right-censored events, the final transaction in the data frame will be the end observation:
    timeFinal = max(data$timestamp)
  }else{
    timeFinal = as.double(observationPeriod[2])
  }
  
  # Collect the index events and select the relevant features:
  indexEvents <- data %>%
    filter(type %in% indexEventSet,
           between(timestamp, timeStart, timeFinal)) %>%
    select(where(not_all_na)) %>%
    mutate(indexID = row_number(),
           indexTime = timestamp) %>%
    select(indexID, timestamp, indexTime, any_of(subjects), any_of(indexCovariates)) %>%
    data.table()
  
  # Collect the outcome events and select the relevant features:
  outcomeEvents <- data %>%
    filter(type %in% outcomeEventSet,
           between(timestamp, timeStart, timeFinal)) %>%
    select(where(not_all_na)) %>%
    mutate(outcomeID = row_number(),
           outcomeTime = timestamp) %>%
    select(outcomeID, timestamp, outcomeTime, any_of(subjects), any_of(outcomeCovariates)) %>%
    data.table()
    
  uncensoredEvents <- outcomeEvents[indexEvents, on = c(subjects, "timestamp"), roll = -Inf] %>%
    filter(!is.na(outcomeID)) %>%
    mutate(timeDiff = outcomeTime - indexTime,
           status = 1) %>%
    select(indexID,
           indexTime,
           outcomeTime,
           timeDiff,
           status,
           any_of(subjects),
           any_of(indexCovariates),
           any_of(outcomeCovariates))
  
  # For right-censored events, we summarise using features from the index event and calculate the 
  # time-to-censorship using the timeFinal from above and the index time:
  rightCensoredEvents <- anti_join(indexEvents, uncensoredEvents, by = c("indexID")) %>%
    mutate(timeDiff = timeFinal - indexTime,
           status = 0) %>%
    select(indexID,
           indexTime,
           timeDiff,
           status,
           any_of(subjects),
           any_of(indexCovariates)) %>%
    distinct()
  
  leftCensoredEvents <- anti_join(outcomeEvents, indexEvents, by = subjects) %>%
    mutate(timeDiff = outcomeTime - timeStart,
           status = 0) %>%
    select(timeDiff,
           outcomeTime,
           status,
           any_of(subjects),
           any_of(outcomeCovariates))
  
  survivalData <- bind_rows(survivalData, rightCensoredEvents, uncensoredEvents, leftCensoredEvents) %>%
    mutate(`Index Event` = toString(indexEventSet),
           `Outcome Event` = toString(outcomeEventSet))
  
  survivalData
}
