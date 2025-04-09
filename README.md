# FinSurvival: A Suite of Large Scale Survival Modeling Tasks from Finance

This repository contains the data being published alongside the paper "FinSurvival: A Suite of Large Scale Survival Modeling Tasks from Finance" in The Journal of Data-centric Machine Learning Research (DMLR), along with code to recreate the figures and results in the paper.

The repository has the following important directories:

## Top-level directory:
The top level of this repository contains two important files (along with, of course, the rest of the directories): "dataLoader.R" and "dataPreprocessing.R". These are used by both the Prediction models and the Classification models to load specific datasets and preprocess a loaded dataset prior to fitting a model. Here's how to use each script:

### dataLoader.R:

### dataPreprocessing.R:


## Data/ directory:
In the "Data/" directory you can find the datasets being published for this work. Most importantly, in the "Data/Survival_Data/" there are 

## Prediction/ directory:
The "Prediction/" directory contains all the code to recreate the results from the paper's first task, FinSurvival.

## Classification/ directory:
The "Classification/" directory contains all the code to recreate the results from the paper's second task, FinSurvival-Classification. 

## Figures_and_Tables/:
The "Figures_and_Tables/" directory contains the code necessary to recreate the tables of results in the paper, along with the heatmaps. It also contains the file assets for the heatmaps.
