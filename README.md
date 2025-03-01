# ML-training
Collection of ML projects using various databases with insights.


## San Francisco Renting Prices 🌉

This repository contains a Python Jupyter notebook that was used to build and evaluate regression models to predict rental prices in San Francisco. The notebook employed various regression techniques, including OLS (Ordinary Least Squares) and Ridge Regression, to predict the price of rentals based on features such as square footage, number of bedrooms, and neighborhood.

### Key Steps:
#### Data Preprocessing:
The data had been cleaned by the provider, and only minor transformations were required. Categorical variables were transformed using one-hot encoding.

#### Exploratory Data Analysis (EDA):
The target variable (price) and independent features were analyzed and visualized. Outliers, multicollinearity, and normality assumptions were checked.

#### Model Training & Evaluation:
OLS regression and Ridge regression models were implemented. k-fold cross-validation was performed to assess model performance. Models were evaluated using R², Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).

#### Diagnostics:
Residuals and other diagnostics were plotted to ensure that the model assumptions were met.

### Used tools: 
Python: NumPy, Pandas, Statsmodels, Scikit Learn, Matplotlib, Seaborn and SciPy.

### Used statistical and modeling methods: 
Descriptive Statistics, Shapiro-Wilk Test, Log Transformation, Feature Engineering, OLS Linear Regression, k-Fold Cross-Validation, Standarization, VIF Verification, Ridge Regression
