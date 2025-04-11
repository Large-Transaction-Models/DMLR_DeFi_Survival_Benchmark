# Data for FinSurvival: A Suite of Large Scale Survival Modeling Tasks from Finance

This directory contains the datasets being published alongside the paper FinSurvival: A Suite of Large Scale Survival Modeling Tasks from Finance in the Datacentric Journal for Machine Learning Research (DMLR).

This includes three subdirectories containing different kinds of data.

1. Survival_Data: The survival data folder contains the bulk of the data, and the most important data for the paper. This is the actual survival data created by the processes outlined in the paper and used in the experiments. The structure of the data in this directory is as follows:
   * Under Survival_Data/ there are sub-directories each representing a possible "index event" as described in the paper.
   * Under Survival_Data/[IndexEvent]/ there are further subdirectories representing possible outcome events for the chosen index event. These are any directories not prefixed with "X_". Additionally, there are directories called "X_train/" and "X_test/". These contain the feature sets for the train and test sets for the chosen index event. In any "X_train/" or "X_test/" directories, there are multiple files that all belong to the same training or testing dataset. They should all be loaded at once and joined together as a single dataframe. The reason they were split into smaller files was to avoid needing to use Github's Large File Storage and potentially exceeding the file sizes allowed.
   * Under Survival_Data/[IndexEvent]/[OutcomeEvent]/" there are two .rds files, "y_train.rds" and "y_test.rds". These represent the target variables for the survival prediction task. Once an index event and outcome event have been selected, the corresponding X_[train/test] and y_[train/test] files can be loaded and joined by the "id" column in order to create the datasets with appropriate targets for the Survival Prediction task in this paper. However, it is recommended that you use the dataLoader.R script to load this data. This script takes care of some additional data cleaning beyond just loading the data.

3. Raw_Transaction_Data_Sample: This folder contains a small sample of the raw transaction data that is used to create the large set of survival data. We only provide a sample of the data for two reasons. First, the full transaction data is much too large to be included in a github repository. Second, we want to keep the raw transaction data private for use in further research.

4. Other_Data: The other data includes any other small dataframes that can be helpful for feature-engineering the survival data.
