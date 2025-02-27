# Profit-Prediction-Model
Profit Prediction Using Regression Models
Abstract
This project focuses on predicting company profits based on their expenditures in R&D, Administration, and Marketing. Various regression models were implemented, including Linear Regression, Lasso Regression, Ridge Regression, and Random Forest Regression. The dataset comprises financial data from 50 companies, and the models were evaluated using key performance metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R²). The objective was to determine the most effective model for accurate profit prediction, with Random Forest Regression emerging as the best performer.
Table of Contents
1. Abstract
2. Table of Contents
3. Introduction
4. Existing Methods
5. Proposed Method and Architecture
6. Methodology
7. Implementation
8. Conclusion
Introduction
Predicting company profits based on financial expenditures is crucial for strategic planning and decision-making. This project leverages machine learning regression models to forecast profits using three key predictors: R&D Spend, Administration Cost, and Marketing Spend. The dataset consists of 50 companies, and the goal is to build a robust predictive model. The study evaluates four regression techniques—Linear Regression, Lasso Regression, Ridge Regression, and Random Forest Regression—to determine the most effective approach.
Existing Methods
Traditional methods, such as Linear Regression, have been widely used for profit prediction. However, they often struggle to capture complex relationships among variables. Regularized regression techniques like Lasso and Ridge help mitigate overfitting by introducing penalty terms, whereas ensemble methods like Random Forest can model non-linear relationships more effectively. By leveraging these advanced methods, the accuracy and generalizability of predictions can be significantly improved.
Proposed Method and Architecture
Proposed Approach
1. Data Preprocessing : Handle missing values, scale features, and split data into training and testing sets.
2. Model Selection: Implement four regression models—Linear Regression, Lasso Regression, Ridge Regression, and Random Forest Regression.
3. Training: Fit each model to the training dataset.
4. Evaluation: Assess model performance using standard regression metrics.
5. Model Comparison: Determine the best-performing model based on evaluation results.

Architecture
- Input Layer: R&D Spend, Administration Cost, and Marketing Spend.
- Regression Models: Linear Regression, Lasso Regression, Ridge Regression, and Random Forest Regression.
- Output Layer: Predicted Profit.
Methodology
Data Preprocessing
- Checked for missing values (none were found).
- Standardized feature values using StandardScaler to ensure uniformity.
- Split dataset into 80% training and 20% testing subsets.

Model Implementation
- Linear Regression: Assumes a linear relationship between independent and dependent variables.
- Lasso Regression: Regularized regression model using L1 penalty to enhance feature selection.
- Ridge Regression: Regularized regression model using L2 penalty to prevent overfitting.
- Random Forest Regression: An ensemble learning method that utilizes multiple decision trees for improved accuracy.

Evaluation Metrics
- Mean Absolute Error (MAE): Measures the average absolute deviation between actual and predicted values.
- Mean Squared Error (MSE): Computes the average squared differences between actual and predicted values.
- Root Mean Squared Error (RMSE): Square root of MSE, offering interpretability in the same units as the target variable.
- R-squared (R²): Represents the proportion of variance in the target variable explained by the model.
Implementation
 Libraries Used
- Pandas: For data manipulation.
- Scikit-learn: For model implementation and evaluation.
- Matplotlib/Seaborn: For data visualization.
 
 Discussion
- Random Forest Regression demonstrated superior performance, achieving the highest accuracy and lowest error.
- Regularized models (Ridge, Lasso) showed improvements over simple Linear Regression, highlighting the importance of penalization techniques.
Conclusion
This study evaluated four regression models for predicting company profits. Random Forest Regression emerged as the most effective approach, providing the highest R² and lowest error values. Regularization techniques such as Ridge Regression also demonstrated their value in enhancing model performance. Future work may involve hyperparameter tuning, feature engineering, or exploring deep learning methods for improved accuracy.
