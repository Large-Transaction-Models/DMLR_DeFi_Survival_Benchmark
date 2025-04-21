# FinSurvival: A Suite of Large Scale Survival Modeling Tasks from Finance

This repository contains the data being published alongside the paper "FinSurvival: A Suite of Large Scale Survival Modeling Tasks from Finance" in The Journal of Data-centric Machine Learning Research (DMLR), along with code to recreate the figures and results in the paper.

The repository has the following important directories:

## Top-level directory:
The top level of this repository contains two important files (along with, of course, the rest of the directories): "dataLoader.R" and "dataPreprocessing.R". These are used by both the Prediction models and the Classification models to load specific datasets and preprocess a loaded dataset prior to fitting a model. Here's how to use each script:

### dataLoader.R:
The dataLoader script is for easily loading a dataset given an index and outcome event. To use it properly, set variables named "indexEvent" and "outcomeEvent" to possible event types. For instance, set "indexEvent = 'Borrow'" and "outcomeEvent = 'Repay'". With these variables in your environment, source the dataLoader file (i.e. run the line "source("../dataLoader.R")". After sourcing this, you will have two datasets loaded in your environment named "train" and "test" which contain the training and testing data, respectively.

### dataPreprocessing.R:
The dataPreprocessing script is used to preprocess train and test sets with various specifications. By default, if you run the dataPreprocessing function and pass it your train and test dataframes, it will scale the data, one-hot encode categorical features, and run PCA, transforming the datasets using the principal components that explain 90% of the variance in the training data. Scaling, one-hot-encoding, and PCA can all be disabled, and the PCA explained variance threshold can be adjusted in the parameters. Additionally, if you want to preprocess the data for the classification task, you can set classificationTask=TRUE and the resulting dataset will have the classfication target instead of the survival prediction targets.

## Data/ directory:
In the "Data/" directory you can find the datasets being published for this work. Most importantly, in the "Data/Survival_Data/" there are three subdirectories. 
* Survival_Data: this is the subdirectory that houses the published survival datasets. They are organized by index and outcome events, and they can be converted to the classification task using the dataPreprocessing function.
* Raw_Transaction_Data_Sample: this directory contains a small sample of the raw transaction data we used to create the survival data.
* Other_Data: this directory was built to contain any other data that might be used to augment the data. It just contains one file which indicates whether each coin type in the data is a stable coin.

## Prediction/ directory:
The "Prediction/" directory contains all the code to recreate the results from the paper's first task, FinSurvival.

## Classification/ directory:
The "Classification/" directory contains all the code to recreate the results from the paper's second task, FinSurvival-Classification. 

## Figures_and_Tables/:
The "Figures_and_Tables/" directory contains the code necessary to recreate the tables of results in the paper, along with the heatmaps. It also contains the file assets for the heatmaps and the Kaplan--Meier survival curves.
