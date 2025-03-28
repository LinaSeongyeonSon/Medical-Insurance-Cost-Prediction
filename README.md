# Medical-Insurance-Cost-Prediction

## Project Overview
This project predicts medical insurance costs based on demographic and health-related factors. Using machine learning techniques, I analyze how variables such as age, BMI, smoking habits, and region impact insurance charges.

## Dataset
The dataset represents the medical insurance costs of indivisuals in the U.S., providing insights into factors influencing insurance charges. (https://www.kaggle.com/datasets/mirichoi0218/insurance/data)
* age: Primary beneficiary's age
* sex: Gender of the insurance holder (female/male)
* bmi: Body mass index (kg/m²), indicating weight relative to height
* children: Number of dependents covered by insurance
* smoker: Smoking status (yes/no)
* region: Residential area in the U.S. (northeast, southeast, southwest, northwest)
* charges: Medical costs billed by insurance

## Libraries Used
* **pandas, numpy:** Data manipulation and numerical operations
* **matplotlib, seaborn:** Data visualization
* **scipy.stats:** Statistical functions (e.g., Pearson correlation)
* **sklearn:** Data preprocessing, model selection, and evaluation (e.g, train/test split, scaling, GridSearchCV, metrics)
* **xgboost, lightgbm:** Gradient boosting models
* **sklearn.linear_model, sklearn.ensemble:** Linear and ensemble regression models

## Data Exploration & Cleaning
* No missing values were found in the dataset.
* Most indivisuals have medical expenses between 1,122 and 13,652.
* Smoking status and BMI were found to be strong indicators of insurance costs.

## Modeling Approach
1. **Preprocessing:** Applies transformations such as One-Hot Encoding for categorical variables and scaling for numerical variables.
2. **Model Selection:** Tested Linear Regression, Polynomial Regression, Random Forest, XG Boost, and LightGBM, and the LightGBM model achieved the best performance.
3. **Hyperparameter Tuning:**
   * Conducted grid search and randomized search to find optimal hyperparameters.
   * Used 5-fold cross-validation with **neg_mean_squared_error** as the evaluation metric.

## Results & Insights
* The **smoker** feature had the highest impact on insurance charges.
* BMI and age also played significant roles in determining costs.
* The **final model** achieved strong predictive performance, reducing error significantly compared to baseline models.

## Future Improvements
* Experiment with additional feature engineering techniques.
* Try deep learning models for comparison.
* Improve interpretability using SHAP values.
