# DMLR_DeFi_Survival_Benchmark - Classification

This project focuses on transforming survival analysis into a classification problem using both traditional machine learning models and deep learning approaches. Specifically, the task is to predict whether a user who has experienced a specific **index event** will trigger a corresponding **outcome event** within a defined time window. The target variable is binary—**"yes"** if the outcome event occurs within this period, and **"no"** otherwise.

To achieve this, we have developed a well-structured survival data pipeline comprising five main sections: **Data Preprocessing**, **Data Processing**, **Model Performance Evaluation and Visualization**, **Classification Models**, and **Deep Learning Models**. This comprehensive pipeline is designed for ease of use, enabling seamless completion of classification tasks across all datasets. Additionally, the pipeline offers high flexibility, allowing users to effortlessly incorporate new classification models and custom functionalities for further analyses or tasks.

## File Descriptions

### Comprehensive Pipeline Documentation
- **`survivalData_pipeline.Rmd`**: Contains detailed documentation of the entire survival data pipeline, including descriptions and implementation details of each function. It serves as a comprehensive reference guide for easy consultation and usage.

### Classification Models
- **`classification_models.R`**: Implements traditional machine learning models including Logistic Regression, Decision Tree, XGBoost, and Elastic Net.

- **`deep_learning_models.R`**: Implements deep learning models, specifically DeepHit and DeepLearningClassifier.

### Python Model Prototypes
- **`deephit_model.py`**: Python prototype for the DeepHit model, callable via functions defined in `deep_learning_models.R` using the `reticulate` package.

- **`deepLearning_classification_model.py`**: Python prototype for the DeepLearningClassifier (Neural Network), also callable through `deep_learning_models.R` using the `reticulate` package.

### Model Evaluation and Visualization
- **`model_evaluation_visual.R`**: Contains helper functions for model evaluation, including calculation and visualization of confusion matrices, accuracy metrics, and various bar graphs. Functions also facilitate dataframe creation, combination, and accuracy analysis.

### Data Processing
- **`data_processing.R`**: Includes functions specifically for data processing tasks, such as SMOTE data balancing and extracting train and test datasets.

### Pipeline Examples
- **`survivalData_pipeline_example.Rmd`**: Provides an illustrative example demonstrating the pipeline's execution process using two datasets, making it clear and easy to understand how to utilize the pipeline effectively and interpret results.

- **`survivalData_pipeline_final.Rmd`**: The most comprehensive implementation of the pipeline, including execution across 16 datasets. This script generates detailed and important accuracy tables like `auc_score.csv` and `model_version.csv`, providing thorough insights into model performance.

## Summary of Functions

- **Data Preprocessing**:
  - `get_classification_cutoff`, `preprocess`

- **Data Processing**:
  - `get_train_test_data`, `smote_data`

- **Model Performance Evaluation and Visualization**:
  - `calculate_model_metrics`, `get_dataframe`, `combine_classification_results`, `accuracy_comparison_plot`, `get_percentage`, `specific_accuracy_table`, `specific_model_version_table`, `combine_accuracy_dataframes`

- **Classification Models**:
  - `logistic_regression`, `decision_tree`, `XG_Boost`, `elastic_net`

- **Deep Learning Models**:
  - `deephit_model`, `deephit_model.py`, `deepLearning_classification_model`, `deepLearning_classification_model.py`