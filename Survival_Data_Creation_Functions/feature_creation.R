library(dplyr)
library(data.table)
library(progress)
library(lubridate)
library(caret)
library(purrr)
library(RcppRoll)


# This is a script to generate all relevant features.
# Note that each functions expect data of the format produced by Aaron's featureCreation.R.
# However, any combination of add feature functions can be run in any order as there are no interdependencies.

# mutate(datetime = as_datetime(timestamp),
#        quarter = floor_date(datetime, unit = "quarter")) %>%
#   mutate(quarter = paste0(year(datetime), " Q", quarter(datetime)))

# Function to make camelCase strings for column names
camel_case <- function(...) {
  strings <- list(...)
  
  # Capitalize the first letter of each string except the first one
  strings <- c(strings[1], lapply(strings[-1], function(x) {
    paste0(toupper(substr(x, 1, 1)), tolower(substr(x, 2, nchar(x))))
  }))
  
  # Concatenate the strings
  paste0(strings, collapse = "")
}


# Function to add columns for count of each type of transaction
# Has cumulative counts and averages for each transaction type
# Number of seconds since first transaction
# Number of seconds since last transaction
add_cumulative_user_history <- function(transactions) {
  transactions <- transactions %>%
    arrange(timestamp) %>%
    group_by(user)
  
  skip_amount_types <- c("swap", "collateral")
  skip_amount_types_coin <- c("swap", "collateral")
  
  # Add isNewUser column: TRUE for first transaction, FALSE otherwise
  transactions <- transactions %>%
    mutate(userIsNew = row_number() == 1)
  
  # Add secondsSinceFirstTransaction and secondsSincePreviousTransaction columns
  transactions <- transactions %>%
    mutate(firstTransactionTime = first(timestamp),
           previousTransactionTime = lag(timestamp),
           userSecondsSinceFirstTransaction = as.numeric(difftime(timestamp, firstTransactionTime, units = "secs")),
           userSecondsSincePreviousTransaction = as.numeric(difftime(timestamp, previousTransactionTime, units = "secs"))) %>%
    select(-firstTransactionTime, -previousTransactionTime)
  
  for (t in unique(transactions$type)) {
    count_col_name <- camel_case("user", t, "count")
    avg_col_name <- camel_case("user", t, "avg", "amount")
    avg_col_nameu <- paste0(camel_case("user", t, "avg", "amount"), "USD")
    avg_col_namee <- paste0(camel_case("user", t, "avg", "amount"), "ETH")
    sum_col_name <- camel_case("user", t, "sum")
    sum_col_nameu <- paste0(camel_case("user", t, "sum"), "USD")
    sum_col_namee <- paste0(camel_case("user", t, "sum"), "ETH")
    
    # First create count column
    transactions <- transactions %>%
      mutate(!!count_col_name := cumsum(type == t))
    
    # Handle regular amount calculations
    if (!(t %in% skip_amount_types)) {
      transactions <- transactions %>%
        mutate(
          # Calculate sum first
          !!sum_col_name := cumsum(ifelse(type == t & !is.na(amount), amount, 0)),
          # Then calculate average, but only when count > 0
          !!avg_col_name := !!sym(sum_col_name) / !!sym(count_col_name)
        )
    }
    
    # Handle USD and ETH calculations
    if (!(t %in% skip_amount_types_coin)) {
      transactions <- transactions %>%
        mutate(
          # USD calculations
          !!sum_col_nameu := cumsum(ifelse(type == t & !is.na(amountUSD), amountUSD, 0)),
          !!avg_col_nameu := !!sym(sum_col_nameu) / !!sym(count_col_name),
          # ETH calculations  
          !!sum_col_namee := cumsum(ifelse(type == t & !is.na(amountETH), amountETH, 0)),
          !!avg_col_namee := !!sym(sum_col_namee) / !!sym(count_col_name)
        )
    }
  }
  
  # Ungroup and arrange the final result
  transactions <- transactions %>%
    ungroup() %>%
    arrange(timestamp)
  
  return(transactions)
}


add_coin_modes <- function(transactions) {
  transactions <- transactions %>%
    arrange(timestamp) %>%
    group_by(user) %>%
    mutate(
      # Initialize counters for reserve and coinType
      userReserveMode = {
        reserve_counts <- list()
        reserve_mode <- NA
        sapply(reserve, function(reserve_val) {
          # Update counts for reserve_val
          reserve_counts[[reserve_val]] <- (reserve_counts[[reserve_val]] %||% 0) + 1
          # Update mode if necessary
          reserve_mode <- names(which.max(unlist(reserve_counts)))
          reserve_mode
        })
      },
      userCoinTypeMode = {
        coinType_counts <- list()
        coinType_mode <- NA
        sapply(coinType, function(coinType_val) {
          # Update counts for coinType_val
          coinType_counts[[coinType_val]] <- (coinType_counts[[coinType_val]] %||% 0) + 1
          # Update mode if necessary
          coinType_mode <- names(which.max(unlist(coinType_counts)))
          coinType_mode
        })
      }
    ) %>%
    ungroup()
  
  return(transactions)
}


# Add days active in window
add_windowed_user_history <- function(transactions) {
  # Convert to data.table for efficient operations
  transactions <- as.data.table(transactions)
  
  # Convert timestamp to Date format if not already
  transactions[, date := as.Date(as.POSIXct(timestamp, origin = "1970-01-01"))]
  
  # Sort by user and date for efficient processing
  setorder(transactions, user, date)
  
  # Initialize columns for active days
  transactions[, `:=`(
    userActiveDaysWeekly = 0,
    userActiveDaysMonthly = 0,
    userActiveDaysYearly = 0
  )]
  
  # Get unique user list
  users <- unique(transactions$user)
  
  # Create and start the progress bar
  pb <- progress_bar$new(
    format = "ActiveDays [:bar] :percent eta: :eta",
    total = length(users),
    clear = FALSE,
    width = 60
  )
  
  # Process each user separately
  for (current_user in users) {
    # Filter data for the current user
    user_data <- transactions[user == current_user]
    
    # Find unique dates for this user
    unique_dates <- unique(user_data$date)
    
    # Calculate rolling counts of active days within 7, 30, and 365 days
    weekly_counts <- sapply(unique_dates, function(d) {
      sum(unique_dates > d - 7 & unique_dates <= d)
    })
    monthly_counts <- sapply(unique_dates, function(d) {
      sum(unique_dates > d - 30 & unique_dates <= d)
    })
    yearly_counts <- sapply(unique_dates, function(d) {
      sum(unique_dates > d - 365 & unique_dates <= d)
    })
    
    # Map these counts back to the main dataset
    transactions[user == current_user, userActiveDaysWeekly := weekly_counts[match(date, unique_dates)]]
    transactions[user == current_user, userActiveDaysMonthly := monthly_counts[match(date, unique_dates)]]
    transactions[user == current_user, userActiveDaysYearly := yearly_counts[match(date, unique_dates)]]
    
    # Update the progress bar
    pb$tick()
  }
  
  transactions <- transactions %>%
    select(-date)
  
  return(transactions)
}


# Classify coins as stable or unstable
add_coin_type <- function(transactions) {
  # coin type
  stableCoins <- read_csv("/data/IDEA_DeFi_Research/Data/Coin_Info/stablecoins.csv", show_col_types=FALSE)
  
  transactions <- transactions %>%
    mutate(coinType = case_when(
      is.na(reserve) ~ NA_character_,
      reserve %in% stableCoins$symbol ~ "Stable",
      TRUE ~ "Non-Stable"
    ))
  
  return(transactions)
}


# Function to add market history features with windowed metrics
# Has windowed counts and averages for each transaction type
add_windowed_market_history <- function(transactions, window_days = 30) {
  # Convert transactions to data.table
  setDT(transactions)
  
  # Ensure the timestamp column is in the correct date-time format
  transactions[, posixTimestamp := as.POSIXct(timestamp)]
  
  # Sort the transactions by timestamp
  setorder(transactions, posixTimestamp)
  
  # Get unique transaction types
  types <- unique(transactions$type)
  
  # Transaction types to skip creating "Amount" columns for
  skip_amount_types <- c("swap", "collateral")
  skip_amount_types_coin <- c("swap", "collateral")
  
  # Generate the column names dynamically and initialize them to 0
  count_cols <- character()
  amount_avg_cols <- character()
  amount_sum_cols <- character()
  amount_avg_colsu <- character()
  amount_sum_colsu <- character()
  amount_avg_colse <- character()
  amount_sum_colse <- character()
  
  for (t in types) {
    # Create "Count" column
    count_col_name <- camel_case("market", t, "count")
    transactions[, (count_col_name) := 0]
    count_cols <- c(count_cols, count_col_name)
    
    # Create "Amount" columns (avg and sum) if the type is not in the skip list
    if (!(t %in% skip_amount_types)) {
      amount_avg_col_name <- camel_case("market", t, "avg", "amount")
      amount_sum_col_name <- camel_case("market", t, "sum")
      transactions[, (amount_avg_col_name) := 0]
      transactions[, (amount_sum_col_name) := 0]
      amount_avg_cols <- c(amount_avg_cols, amount_avg_col_name)
      amount_sum_cols <- c(amount_sum_cols, amount_sum_col_name)
    }
    if (!(t %in% skip_amount_types_coin)) {
      amount_avg_col_nameu <- paste0(camel_case("market", t, "avg", "amount"), "USD")
      amount_sum_col_nameu <- paste0(camel_case("market", t, "sum"), "USD")
      amount_avg_col_namee <- paste0(camel_case("market", t, "avg", "amount"), "ETH")
      amount_sum_col_namee <- paste0(camel_case("market", t, "sum"), "ETH")
      
      transactions[, (amount_avg_col_nameu) := 0]
      transactions[, (amount_sum_col_nameu) := 0]
      transactions[, (amount_avg_col_namee) := 0]
      transactions[, (amount_sum_col_namee) := 0]
      
      amount_avg_colsu <- c(amount_avg_colsu, amount_avg_col_nameu)
      amount_sum_colsu <- c(amount_sum_colsu, amount_sum_col_nameu)
      amount_avg_colse <- c(amount_avg_colse, amount_avg_col_namee)
      amount_sum_colse <- c(amount_sum_colse, amount_sum_col_namee)
    }
  }
  
  n <- nrow(transactions)
  window_start <- 1
  type_counts <- setNames(numeric(length(types)), types)
  type_amounts <- setNames(numeric(length(types)), types)
  type_amountsu <- setNames(numeric(length(types)), types)
  type_amountse <- setNames(numeric(length(types)), types)
  
  # Precompute window length in seconds
  window_length <- window_days * 24 * 60 * 60
  
  # Initialize the progress bar
  pb <- progress_bar$new(
    format = "  Adding windowed market history: [:bar] :percent in :elapsed",
    total = n, clear = FALSE, width = 60
  )
  
  for (i in 1:n) {
    current_time <- transactions$posixTimestamp[i]
    window_end_time <- current_time - window_length
    
    # Remove transactions outside the window
    while (window_start < i && transactions$posixTimestamp[window_start] < window_end_time) {
      old_type <- transactions$type[window_start]
      type_counts[old_type] <- type_counts[old_type] - 1
      if (!(old_type %in% skip_amount_types)){
        type_amounts[old_type] <- type_amounts[old_type] - transactions$amount[window_start]
      }
      if (!(old_type %in% skip_amount_types_coin)){
        type_amountsu[old_type] <- type_amountsu[old_type] - transactions$amountUSD[window_start]
        type_amountse[old_type] <- type_amountse[old_type] - transactions$amountETH[window_start]
      }
      window_start <- window_start + 1
    }
    
    # Add the current transaction
    current_type <- transactions$type[i]
    type_counts[current_type] <- type_counts[current_type] + 1
    if (!(current_type %in% skip_amount_types)){
      type_amounts[current_type] <- type_amounts[current_type] + transactions$amount[i]
    }
    if (!(current_type %in% skip_amount_types_coin)){
      type_amountsu[current_type] <- type_amountsu[current_type] + transactions$amountUSD[i]
      type_amountse[current_type] <- type_amountse[current_type] + transactions$amountETH[i]
    }
    
    # Update type counts for this row
    set(transactions, i, count_cols, as.list(type_counts))
    
    # Update type amounts for this row (only for types not in skip_amount_types)
    set(transactions, i, amount_avg_cols, as.list(type_amounts[!(types %in% skip_amount_types)] / pmax(type_counts[!(types %in% skip_amount_types)], 1)))
    set(transactions, i, amount_sum_cols, as.list(type_amounts[!(types %in% skip_amount_types)]))
    set(transactions, i, amount_avg_colsu, as.list(type_amountsu[!(types %in% skip_amount_types_coin)] / pmax(type_counts[!(types %in% skip_amount_types_coin)], 1)))
    set(transactions, i, amount_sum_colsu, as.list(type_amountsu[!(types %in% skip_amount_types_coin)]))
    set(transactions, i, amount_avg_colse, as.list(type_amountse[!(types %in% skip_amount_types_coin)] / pmax(type_counts[!(types %in% skip_amount_types_coin)], 1)))
    set(transactions, i, amount_sum_colse, as.list(type_amountse[!(types %in% skip_amount_types_coin)]))
    
    # Update the progress bar
    pb$tick()
  }
  
  transactions <- as.data.frame(transactions)
  transactions$posixTimestamp <- NULL
  
  return(transactions)
}


# Create time features by day, week, and add in their circular representation as well
# Calculates the sin/cos of the day/week values so that 11:59PM and 12:00AM are close together
create_time_features <- function(DT) {
  # Ensure we're working with a data.table
  if (!is.data.table(DT)) {
    DT <- as.data.table(DT)
  }
  DT[, posixTimestamp := as.POSIXct(timestamp)]
  
  # Basic time features
  DT[, `:=`(
    timeOfDay = as.numeric(format(posixTimestamp, "%H")) + 
      as.numeric(format(posixTimestamp, "%M"))/60 +
      as.numeric(format(posixTimestamp, "%S"))/3600,
    dayOfWeek = as.numeric(format(posixTimestamp, "%u")),  # 1-7, Monday is 1
    dayOfMonth = as.numeric(format(posixTimestamp, "%d")),
    dayOfYear = as.numeric(format(posixTimestamp, "%j")),
    quarter = quarter(posixTimestamp)
  )]
  
  # Calculate dayOfQuarter
  DT[, dayOfQuarter := dayOfYear - c(0, 90, 181, 273)[quarter] + 1]
  
  # Cyclical features using sine and cosine
  # For timeOfDay (24-hour cycle)
  DT[, `:=`(
    sinTimeOfDay = sin(2 * pi * timeOfDay / 24),
    cosTimeOfDay = cos(2 * pi * timeOfDay / 24)
  )]
  
  # For dayOfWeek (7-day cycle)
  DT[, `:=`(
    sinDayOfWeek = sin(2 * pi * dayOfWeek / 7),
    cosDayOfWeek = cos(2 * pi * dayOfWeek / 7)
  )]
  
  # For dayOfMonth (28-31 day cycle, using 30 as average)
  DT[, `:=`(
    sinDayOfMonth = sin(2 * pi * dayOfMonth / 30),
    cosDayOfMonth = cos(2 * pi * dayOfMonth / 30)
  )]
  
  # For dayOfQuarter (90-92 day cycle, using 91 as average)
  DT[, `:=`(
    sinDayOfQuarter = sin(2 * pi * dayOfQuarter / 91),
    cosDayOfQuarter = cos(2 * pi * dayOfQuarter / 91)
  )]
  
  # For dayOfYear (365/366 day cycle)
  DT[, `:=`(
    sinDayOfYear = sin(2 * pi * dayOfYear / 365),
    cosDayOfYear = cos(2 * pi * dayOfYear / 365)
  )]
  
  # For quarter (4 quarters in a year)
  DT[, `:=`(
    sinQuarter = sin(2 * pi * quarter / 4),
    cosQuarter = cos(2 * pi * quarter / 4)
  )]
  
  DT[, isWeekend := as.logical(ifelse(dayOfWeek %in% c(6, 7), 1, 0))]
  
  DT <- as.data.frame(DT)
  DT$posixTimestamp <- NULL
  return(DT)
}


# Create some simple log of amounts features
add_log_features <- function(transactions) {
  transactions[["logAmount"]] <- log1p(transactions[["amount"]])
  transactions[["logAmountUSD"]] <- log1p(transactions[["amountUSD"]])
  transactions[["logAmountETH"]] <- log1p(transactions[["amountETH"]])
  
  return(transactions)
}


# Create PCA features on numeric columns in the data.
#   An important thing to note is that not all transactions share the same set of numeric vars.
#   As a result, there will be specific groups of vars considered for each transaction type 
add_PCA_features = function(transactions, rawTransactions){
  
  # Get all unique types from the transactions dataframe
  transaction_types = unique(transactions$type)
  
  # For each type, perform PCA and cast it back onto the dataframe for those transactions
  for(type in transaction_types){
    print(type)
    
    # In implementing this, I noticed that events of type "collateral" have no numeric information associated with them. 
    # As a result, they are simply skipped, since PCA cannot help us in that case
    if(type == "collateral"){
      next
    }
    
    # Get the type-specific data
    type_data = transactions[transactions$type == type, ]
    
    # Keep only base numeric features from rawTransactions
    features_to_keep = names(rawTransactions)[sapply(rawTransactions, is.numeric)]
    features_to_keep <- intersect(features_to_keep, names(type_data))
    type_data = type_data[, features_to_keep, drop = FALSE]
    
    # Remove timestamp if it exists
    if("timestamp" %in% names(type_data)) {
      type_data$timestamp <- NULL
    }
    
    # Extract numeric data
    type_data_num = type_data %>% 
      select(where(function(col) is.numeric(col) && !all(is.na(col))))
    type_data_non_num = type_data %>% select(where(function(col) !is.numeric(col)))
    
    # Perform PCA
    pca_model = preProcess(type_data_num, method = "pca", thresh = 0.95)  # `thresh` can control variance explained
    pca_transform = predict(pca_model, type_data_num)
    
    # Add type as suffix to PCA columns
    colnames(pca_transform) <- paste0( type, colnames(pca_transform))
    
    # Bind the PCA columns back to the original transactions dataframe
    type_indices = which(transactions$type == type)
    transactions[type_indices, colnames(pca_transform)] <- pca_transform
  }
  return(transactions)
}


# This adds various interaction terms to the dataset.
#   Most of them build off the work Sean did in a notebook, finding that interactions between AvgPriorAmount and priorEntries

# NOTE: This must be run after the other feature creation functions have been run, otherwise the avg and count columns will not exist yet in the dataframe.

add_interaction_features = function(transactions){
  
  # Get all the column names of the prior amounts (count), and AvgPriorAmount (AvgAmount)
  count_features = grep("Count$", names(transactions), value=TRUE)
  avg_amount_features = grep("AvgAmount$", names(transactions), value=TRUE)
  
  # Loop through each one of these features, and add an interaction term between them
  
  for(count_col in count_features){
    for(avg_amount in avg_amount_features){
      
      
      # Only want to add interaction between like count and avg amount terms
      count_prefix = sub("Count$", "", count_col)
      avg_prefix = sub("AvgAmount", "", avg_amount)
      
      if( count_prefix == avg_prefix){
        # Create the name of the new feature
        new_feature_name = paste(count_col, avg_amount, sep = "_x_")
        
        # Calculate and add the interaction term to transactions
        transactions[new_feature_name] = transactions[count_col] * transactions[avg_amount]
      }
    }
    
    # Another important feature identified by Sean was the count X log_amountUSD, so they are added here
    feature_name = paste(count_col, "logAmountUSD", sep = "_x_")
    transactions[feature_name] = transactions[count_col] * transactions$log_amountUSD
    
    
  }
  return (transactions)
}


add_crypto_prices <- function(transactions) {
  # Convert transactions to data.table if it isn't already
  transactions <- as.data.table(transactions)
  
  # Create temporary date columns for joining
  transactions[, `:=`(
    posixTimestamp = as.POSIXct(timestamp, origin = "1970-01-01"),
    date = as.Date(as.POSIXct(timestamp, origin = "1970-01-01"))
  )]
  
  # Initialize an empty data.table to store crypto prices
  combined_prices <- NULL
  
  # Get list of files in directory
  files <- list.files("/data/IDEA_DeFi_Research/Data/Market_Watch_Data/Cryptos", 
                      pattern = "*.csv", 
                      full.names = TRUE)
  
  for (file in files) {
    # Extract filename without extension for the column name
    file_name <- tools::file_path_sans_ext(basename(file))
    # Convert filename to camelCase
    col_name <- paste0(tolower(substr(file_name, 1, 1)), substr(file_name, 2, nchar(file_name)))
    
    # Read in the CSV file as data.table
    dt <- fread(file)
    
    # Check for possible date column names
    date_col <- grep("day|date", names(dt), ignore.case = TRUE, value = TRUE)[1]
    price_col <- grep("price", names(dt), ignore.case = TRUE, value = TRUE)[1]
    
    # Verify that columns are found
    if (is.null(date_col) || is.null(price_col)) {
      stop(paste("File", file, "must contain a date and price column."))
    }
    
    # Rename columns to standard names
    setnames(dt, c(date_col, price_col), c("date", "price"))
    
    # Convert date column to Date type (handling ISO 8601 format)
    dt[, date := as.Date(sub("T.*", "", date))]
    
    # Create the lagged price column
    dt[order(date), (col_name) := shift(price, 1)]
    
    # Keep only date and the new lagged price column
    dt <- dt[, .(date, get(col_name))]
    setnames(dt, "V2", col_name)
    
    # Merge with combined prices
    if (is.null(combined_prices)) {
      combined_prices <- dt
    } else {
      combined_prices <- merge(combined_prices, dt, by = "date", all = TRUE)
    }
  }
  
  # Sort combined prices by date
  setorder(combined_prices, date)
  combined_prices <- unique(combined_prices, by = "date")
  
  # Join the prices with transactions
  result <- merge(transactions, combined_prices, by = "date", all.x = TRUE)
  
  # Remove temporary date columns
  result[, c("date", "posixTimestamp") := NULL]
  
  # Convert to data.frame and return
  return(as.data.frame(result))
}


add_YFinance_data <- function(transactions, lagged=FALSE) {
  transactions <- as.data.table(transactions)

  transactions[, date := as.Date(as.POSIXct(timestamp, origin = "1970-01-01"))]
  
  yfinance <- fread("/data/IDEA_DeFi_Research/Data/Market_Watch_Data/YFinance/market_data_2020-11-28_to_2024-05-25.csv")
  cols_to_rename <- setdiff(names(yfinance), "date")
  setnames(yfinance, old = cols_to_rename, new = paste0("exo", toupper(substring(cols_to_rename, 1, 1)), substring(cols_to_rename, 2)))
  
  # lag
  if (lagged) {
    yfinance[, date := as.Date(date) + 1]
  }
  
  result <- transactions[yfinance, on = "date", nomatch = 0]
  result[, date := NULL]
  
  return(result)
}


rawTransactions <- read_rds("/data/IDEA_DeFi_Research/Data/Lending_Protocols/Aave/V2/Mainnet/transactions.rds") 

# Data cleaning
rawTransactions <- rawTransactions[!(rawTransactions$type == "liquidation" & (is.na(rawTransactions$collateralAmount) | is.na(rawTransactions$principalAmount))), ]
rawTransactions <- rawTransactions[!(rawTransactions$type == "flashLoan"), ]
rawTransactions <- rawTransactions %>%
  mutate(user = ifelse(is.na(user), onBehalfOf, user)) %>%
  mutate(
    reserve = if_else(type == "liquidation", principalReserve, reserve),
    amount = if_else(type == "liquidation", principalAmount, amount),
    amountUSD = if_else(type == "liquidation", principalAmountUSD, amountUSD),
    amountETH = if_else(type == "liquidation", principalAmountETH, amountETH)
  ) %>%
  mutate(priceInUsd = if_else(is.na(priceInUsd), amountUSD / amount, priceInUsd)) %>%
  rename(priceInUSD = priceInUsd) %>%
  select(-userAlias, -onBehalfOfAlias, -liquidatorAlias, -onBehalfOf, 
         -principalReserve, -principalAmount, -principalAmountUSD, -principalAmountETH, 
         -version, -deployment, -target)

# Add features using the above functions
transactions <- add_coin_type(transactions)
transactions <- add_coin_modes(transactions)
transactions <- add_cumulative_user_history(transactions)
transactions <- add_windowed_user_history(transactions)
transactions <- add_windowed_market_history(transactions)
transactions <- create_time_features(transactions)
transactions <- add_log_features(transactions)

# Fill NAs with 0 for columns containing "AvgAmount" in their name and timeSincePreviousTransaction
transactions <- transactions %>%
  mutate(across(c(contains("AvgAmount"), "userSecondsSincePreviousTransaction"), ~ replace_na(., 0)))

# Convert relevant cols to factors
cols_to_convert <- names(transactions)[sapply(transactions, function(col) is.logical(col) || is.character(col))]
cols_to_convert <- setdiff(cols_to_convert, c("user", "id", "liquidator"))
transactions[cols_to_convert] <- lapply(transactions[cols_to_convert], as.factor)

saveRDS(transactions, file = "/data/IDEA_DeFi_Research/Data/Lending_Protocols/Aave/V2/Mainnet/Experimental/transactions_user_market_time.rds")

# Create exogenous datasets
transactions <- NULL
gc()
transactions <- readRDS("/data/IDEA_DeFi_Research/Data/Lending_Protocols/Aave/V2/Mainnet/Experimental/transactions_user_market_time.rds")
transactions <- add_YFinance_data(transactions, TRUE)
saveRDS(transactions, file = "/data/IDEA_DeFi_Research/Data/Lending_Protocols/Aave/V2/Mainnet/Experimental/transactions_user_market_time_exoLagged.rds")
