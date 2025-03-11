# ML-training
Collection of ML projects using various databases with insights.


## San Francisco Renting Prices 🌉 [Regression]

This project contains a Python Jupyter notebook that was used to build and evaluate regression models to predict rental prices in San Francisco. The notebook employed various regression techniques, including OLS (Ordinary Least Squares) and Ridge Regression, to predict the price of rentals based on features such as square footage, number of bedrooms, and neighborhood.

### Key Steps:
#### Data Preprocessing:
The data had been cleaned by the provider, and only minor transformations were required. Categorical variables were transformed using one-hot encoding.

#### Exploratory Data Analysis (EDA):
The target variable (price) and independent features were analyzed and visualized. Outliers, multicollinearity, and normality assumptions were checked.

#### Model Training & Evaluation:
OLS regression and Ridge regression models were implemented. k-fold cross-validation was performed to assess model performance. Models were evaluated using R², Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).

#### Diagnostics:
Residuals and other diagnostics were plotted to ensure that the model assumptions were met.

### 🛠️ Used tools: 
Python: NumPy, Pandas, Statsmodels, Scikit Learn, Matplotlib, Seaborn and SciPy.

### Used statistical and modeling methods: 
Descriptive Statistics, Shapiro-Wilk Test, Log Transformation, Feature Engineering, OLS Linear Regression, k-Fold Cross-Validation, Standarization, VIF Verification, Ridge Regression




## Breast Cancer Classification 🏥 [Classification]
This project contains a Python Jupyter notebook used to build and evaluate classification models for diagnosing breast cancer as malignant or benign. The notebook applies K-Nearest Neighbors (KNN) classification, with feature selection and hyperparameter tuning, to improve recall and minimize false negatives.

### Key Steps
#### Data Preprocessing:
The dataset was cleaned by removing unnecessary columns. The target variable (diagnosis) was encoded as binary (1 = Malignant, 0 = Benign). Features were standardized using StandardScaler to ensure uniform distribution.

#### Exploratory Data Analysis (EDA):
The correlation between features was analyzed to detect multicollinearity. A heatmap was generated to identify highly correlated features (correlation > 0.90). Class distribution was checked for potential imbalance.

#### Model Training & Evaluation:
A baseline KNN model was trained on all features. 
Key evaluation metric - Recall (Sensitivity): How many actual malignant cases were correctly identified.

#### Feature Selection & Dimensionality Reduction:
Removing Highly Correlated Features: The reduced model maintained similar recall and accuracy but was simpler and more interpretable.
Principal Component Analysis (PCA) (Discarded): PCA was applied to reduce dimensionality while preserving variance. PCA was ultimately rejected, as it did not improve recall and made interpretation difficult.

#### Hyperparameter Tuning with GridSearchCV:
Best Parameters Found: {'metric': 'manhattan', 'n_neighbors': 3, 'weights': 'uniform'}
Despite tuning, false negatives remained, posing a diagnostic risk.

### 🛠️ Used tools: 
Python: NumPy, Pandas, Scikit Learn, Matplotlib, Seaborn

📊 Used Statistical and Modeling Methods:
Descriptive Statistics, Feature Engineering, Correlation Analysis, Standardization (StandardScaler), K-Nearest Neighbors (KNN) Classification, GridSearchCV for Hyperparameter Tuning, Confusion Matrix Analysis, PCA (Principal Component Analysis) (Tested but discarded).
