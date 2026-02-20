# Term Deposit Subscription Prediction (Bank Marketing)

## Overview
This project focuses on predicting whether a bank customer will subscribe to a term deposit based on marketing campaign data. The task leverages machine learning classifiers along with explainable AI to derive meaningful, interpretable insights from the model predictions.

## Objective
To build and evaluate predictive models that can accurately forecast whether a client will subscribe to a term deposit product, using the bank's historical marketing data.

## My Approach
1. **Data Cleaning**: Removed irrelevant or redundant features, managed categorical columns, and dealt with class imbalance.
2. **Exploratory Data Analysis (EDA)**: Visualized relationships between key features and the target variable.
3. **Preprocessing**:
   - Encoded categorical variables
   - Addressed class imbalance using oversampling
   - Scaled numerical features for consistency
4. **Model Building**:
   - Trained Logistic Regression and Random Forest models
   - Evaluated model performance using accuracy, confusion matrix, and ROC curves
5. **Model Interpretability**:
   - Used SHAP (SHapley Additive exPlanations) to explain feature importance and model decisions

## Key Insights
- **Duration** of the last contact is the strongest predictor of term deposit subscription.
- Other important features include:
  - **poutcome**: outcome of the previous campaign
  - **contact**: communication type (e.g., cellular, telephone)
  - **housing**: whether the customer has a housing loan
- High values for key features (e.g., long duration, successful previous outcome) significantly increase the likelihood of a positive prediction.
- Low or missing feature values tend to reduce the model's confidence.


# Customer Segmentation Using Unsupervised Learning

## Overview
This project applies unsupervised machine learning techniques to segment customers based on their purchasing behavior. By identifying distinct customer groups, businesses can tailor their marketing strategies and offerings to better serve each segment.

## Objective
To identify meaningful customer segments from behavioral data using clustering algorithms and visualize these segments for actionable business insights.

## My Approach
1. **Data Cleaning and Preprocessing**
   - Removed missing or duplicate records
   - Scaled features using standardization for distance-based clustering
2. **Exploratory Data Analysis (EDA)**
   - Explored distributions and correlations between features
3. **Clustering**
   - Applied the KMeans algorithm to segment customers
   - Determined the optimal number of clusters using the Elbow Method and Silhouette Score
4. **Dimensionality Reduction and Visualization**
   - Used Principal Component Analysis (PCA) and t-SNE for visualizing clusters in 2D space

## Key Insights
- Distinct customer clusters were identified, each representing a group with shared characteristics such as high income, low spending, or younger age profiles.
- Visualization of clusters revealed patterns that can help target marketing efforts more precisely.
- Example strategies include targeting high-income low-spending customers with exclusive offers or engaging younger segments through digital channels.


# Energy Consumption Time Series Forecasting
Forecasting household electricity usage using ARIMA, Prophet, and XGBoost models.

## Overview
This project focuses on short-term energy consumption forecasting using historical household power consumption data. By leveraging classical statistical methods and machine learning, we explore seasonal patterns and make accurate daily forecasts. 

## Objective
- Predict daily household energy usage based on historical patterns.
- Compare the performance of ARIMA, Prophet, and XGBoost models.
- Evaluate models based on metrics like MAE, MSE, and RMSE.

## My Approach
1. **Data Cleaning & Preparation**  
   - Converted datetime fields
   - Handled missing values
   - Resampled to daily averages

2. **Exploratory Data Analysis**  
   - Visualized seasonal trends and volatility
   - Decomposed time series into trend, seasonality, and residuals

3. **Stationarity Testing**  
   - Conducted Augmented Dickey-Fuller (ADF) test
   - Applied differencing to achieve stationarity

4. **Modeling**  
   - Tuned ARIMA parameters using grid search on (p,d,q)
   - Trained Prophet model with additive seasonality
   - Used XGBoost with engineered features and time-based cross-validation
5. **Evaluation**  
   - Compared models using MAE, MSE, and RMSE
   - Visualized actual vs. predicted values for all models
     
## Key Insights
- **ARIMA (6,1,7)** model achieved the lowest RMSE (**0.316**), outperforming Prophet and XGBoost.
- **Prophet** handled seasonality components well and offered high interpretability.
- **XGBoost** leveraged time-based features effectively, producing competitive forecasts.
- Clear **weekly and monthly seasonality** patterns were identified in the data.
- Feature engineering and proper resampling significantly improved model performance.

##  Results Summary

| Model   | MAE    | MSE     | RMSE    |
|---------|--------|---------|---------|
| ARIMA   | 0.224  | 0.100   | 0.316 |
| Prophet | 0.240  | 0.106   | 0.326   |
| XGBoost | 0.245  | 0.105   | 0.325   |


# Loan Default Risk Prediction with Business Cost Optimization

## Overview

This project focuses on predicting the likelihood of loan default and optimizing classification thresholds to minimize business cost. Instead of relying on conventional performance metrics alone (like accuracy or AUC), this project integrates a cost-sensitive approach where financial consequences of false positives and false negatives are explicitly considered. 

## Objective
The main objective is twofold:
1. Predict the likelihood of a loan applicant defaulting using machine learning.
2. Optimize the decision threshold to minimize the total cost to the business from false positives and false negatives.

## My Approach

1. **Data Preprocessing**
   - **Missing Values**: Handled any missing or inconsistent entries.
   - **Categorical Encoding**:
      - For **Logistic Regression**, applied one-hot encoding.
     - For **CatBoost**, used raw categorical variables (CatBoost handles these internally).
   - **Feature Scaling**:
      - Applied standardization to numerical features for Logistic Regression.
     - No scaling needed for CatBoost.

2. **Model Training**
   
     Trained and compared two models:
   - **Logistic Regression**
      - Applied after preprocessing (encoding + scaling).
     - Offers interpretability and serves as a baseline.
   - **CatBoostClassifier**
      - Chosen for its performance on tabular data and native support for categorical features.
     - Tuned hyperparameters and trained on the same dataset.

3. **Business Cost Function** 
   - Defined a cost function based on business logic:
     - **False Positive (FP)**: $1,000 (missed opportunity from rejecting a good applicant)
     - **False Negative (FN)**: $10,000 (loss from approving a defaulter)

4. **Threshold Optimization**
   - Evaluated thresholds from 0 to 1 using the predicted probabilities.
   - Calculated total business cost for each threshold using:<br>
       total_cost = FP × cost_fp + FN × cost_fn
   - Selected the threshold that minimized the total cost.


5. **Model Evaluation**
   - Confusion matrix and classification metrics (precision, recall, F1-score) were analyzed at the optimized threshold.
   - ROC AUC was used to confirm model discrimination performance.
   - A cost vs. threshold plot was created to visually highlight the impact of threshold selection.

## Key Insights
- **Default Threshold (0.50):**
   - Cost: $2,369,000

- **Optimized Threshold (0.21):**
   - Cost: $1,802,000
   - Cost Saved: $567,000

- Despite some increase in false positives, the large reduction in false negatives significantly reduced overall business loss.
- Optimizing threshold based on cost delivers better real-world outcomes than relying on standard metrics alone.

