# random_survival_forest (ranger implementation)
# - Inputs: train, test data.frames with columns `timeDiff` (seconds) and `status` (0/1)
# - Optional: `features` character vector to subset predictors
# - Output: list(predictions, model, time_grid, survival_curves, chf_curves)
# - Predictions: risk-like score = final cumulative hazard (bigger = worse) → use hazard=TRUE in your C-index

random_survival_forest <- function(train, test, features = NULL,
                                   num.trees = 100,        # fast default for 150k x 130; bump for final runs
                                   mtry = NULL,            # if NULL → ~sqrt(p)
                                   min.node.size = 300,    # larger = shallower/faster trees
                                   sample.fraction = 0.5,  # subsample per tree for speed; set to 1.0 for standard RF
                                   replace = TRUE,
                                   respect.unordered.factors = c("order", "partition")[1],
                                   num.threads = max(1, parallel::detectCores() - 1),
                                   seed = 42,
                                   impute = TRUE) {
  
  # --- pkgs ---
  if (!requireNamespace("ranger", quietly = TRUE)) install.packages("ranger")
  if (!requireNamespace("survival", quietly = TRUE)) install.packages("survival")
  library(ranger); library(survival)
  
  set.seed(seed)
  
  
  if (!is.null(features)) {
    keep <- intersect(features, colnames(train))
    keep <- union(keep, c("timeDiff", "status"))
    train <- train[, intersect(keep, colnames(train)), drop = FALSE]
    test  <- test[,  intersect(keep, colnames(test)),  drop = FALSE]
  }
  
  # --- light type cleanup: chars/logicals -> factors (ranger handles factors efficiently) ---
  to_factor <- function(df) {
    for (nm in names(df)) {
      if (is.logical(df[[nm]]) || is.character(df[[nm]])) {
        df[[nm]] <- factor(df[[nm]])
      }
    }
    df
  }
  train <- to_factor(train)
  test  <- to_factor(test)
  
  # --- align factor levels in test to match train ---
  common <- intersect(names(train), names(test))
  for (nm in common) {
    if (is.factor(train[[nm]])) {
      test[[nm]] <- factor(as.character(test[[nm]]), levels = levels(train[[nm]]))
    }
  }
  
  # --- fast median/mode imputation to avoid row drops in ranger ---
  if (isTRUE(impute)) {
    impute_one <- function(x) {
      if (is.numeric(x)) {
        if (anyNA(x)) x[is.na(x)] <- stats::median(x, na.rm = TRUE)
      } else if (is.factor(x)) {
        if (anyNA(x)) {
          tab <- table(x, useNA = "no")
          if (length(tab)) {
            mode_lvl <- names(tab)[which.max(tab)]
            x[is.na(x)] <- mode_lvl
          } else {
            # empty factor edge-case: set to first level if exists
            if (length(levels(x)) > 0) x[is.na(x)] <- levels(x)[1]
          }
        }
      }
      x
    }
    # do not impute the targets
    target_cols <- c("timeDiff", "status")
    for (nm in setdiff(names(train), target_cols)) train[[nm]] <- impute_one(train[[nm]])
    for (nm in setdiff(names(test),  target_cols)) test[[nm]]  <- impute_one(test[[nm]])
  }
  
  # --- build formula ---
  predictors <- setdiff(colnames(train), c("timeDiff", "status"))
  if (length(predictors) == 0) stop("No predictor columns found after subsetting.")
  if (is.null(mtry)) mtry <- max(1, floor(sqrt(length(predictors))))
  form <- as.formula(paste0("Surv(timeDiff/86400, status) ~ ", paste(predictors, collapse = " + ")))
  
  # --- fit ---
  fit <- ranger::ranger(
    formula                     = form,
    data                        = train,
    num.trees                   = num.trees,
    mtry                        = mtry,
    min.node.size               = min.node.size,
    splitrule                   = "logrank",
    respect.unordered.factors   = respect.unordered.factors, # "order" is fast; "partition" is unbiased but slower
    sample.fraction             = sample.fraction,
    replace                     = replace,
    num.threads                 = num.threads,
    seed                        = seed,
    keep.inbag                  = FALSE
  )
  
  # --- predict ---
  pred <- predict(fit, data = test, type = "response")
  
  # Risk-like score for C-index (larger = worse): final cumulative hazard per row
  chf_mat <- pred$chf
  risk_score <- as.numeric(chf_mat[, ncol(chf_mat)])
  
  # try to grab time grid; different ranger versions store different attrs
  tg <- attr(chf_mat, "unique.death.times")
  if (is.null(tg)) tg <- attr(chf_mat, "time.interest")
  
  # return in the same spirit as your other scripts
  return(list(
    predictions     = risk_score,            # pass hazard=TRUE to concordanceIndex(...)
    model           = fit,
    time_grid       = tg,                    # vector of times (days)
    survival_curves = pred$survival,         # matrix [n_test x length(time_grid)]
    chf_curves      = chf_mat                # matrix [n_test x length(time_grid)]
  ))
}
