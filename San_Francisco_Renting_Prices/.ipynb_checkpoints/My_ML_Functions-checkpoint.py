import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score as r2, mean_absolute_error as mae, mean_squared_error as mse
from sklearn.linear_model import Ridge


def reg_cross_validate_model(X, y, n_splits=5):
    """
    Performs k-fold cross-validation for an OLS regression model using statsmodels.
    Calculates R², MAE, and RMSE for both training and test sets in each fold.
    Then trains a final model on the entire dataset, prints its summary as well as MAE and RMSE.
    
    Parameters:
        X (DataFrame): Feature matrix (without constant term).
        y (Series): Target variable.
        n_splits (int): Number of folds for cross-validation.
        
    Returns:
        final_model: The OLS model fitted on the entire dataset.
    """
    # Add constant to the entire feature set (for the intercept)
    X_with_const = sm.add_constant(X)
    
    # Initialize KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=2023)
    
    # Lists to store metrics for each fold
    train_r2_list, train_mae_list, train_rmse_list = [], [], []
    test_r2_list, test_mae_list, test_rmse_list = [], [], []
    
    # Cross-validation loop
    for fold, (train_index, test_index) in enumerate(kf.split(X_with_const), start=1):
        # Split data into training and test sets for the current fold
        X_train, X_test = X_with_const.iloc[train_index], X_with_const.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Fit the OLS model on training data
        model = sm.OLS(y_train, X_train).fit()
        
        # Generate predictions for training and test sets
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics for the training set
        r2_train = r2(y_train, y_train_pred)
        mae_train = mae(y_train, y_train_pred)
        rmse_train = np.sqrt(mse(y_train, y_train_pred))
        
        # Calculate metrics for the test set
        r2_test = r2(y_test, y_test_pred)
        mae_test = mae(y_test, y_test_pred)
        rmse_test = np.sqrt(mse(y_test, y_test_pred))
        
        # Append computed metrics to the lists
        train_r2_list.append(r2_train)
        train_mae_list.append(mae_train)
        train_rmse_list.append(rmse_train)
        
        test_r2_list.append(r2_test)
        test_mae_list.append(mae_test)
        test_rmse_list.append(rmse_test)
        
        # Print metrics for the current fold
        print(f"Fold {fold}:")
        print(f"  Training -> R²: {r2_train:.3f}, MAE: {mae_train:.3f}, RMSE: {rmse_train:.3f}")
        print(f"  Test     -> R²: {r2_test:.3f}, MAE: {mae_test:.3f}, RMSE: {rmse_test:.3f}")
        print("-" * 50)
    
    # Print average metrics across all folds
    print("Summary of Cross-Validation Metrics:")
    print(f"Average Training R²: {np.mean(train_r2_list):.3f} ± {np.std(train_r2_list):.3f}")
    print(f"Average Training MAE: {np.mean(train_mae_list):.3f} ± {np.std(train_mae_list):.3f}")
    print(f"Average Training RMSE: {np.mean(train_rmse_list):.3f} ± {np.std(train_rmse_list):.3f}")
    print()
    print(f"Average Test R²: {np.mean(test_r2_list):.3f} ± {np.std(test_r2_list):.3f}")
    print(f"Average Test MAE: {np.mean(test_mae_list):.3f} ± {np.std(test_mae_list):.3f}")
    print(f"Average Test RMSE: {np.mean(test_rmse_list):.3f} ± {np.std(test_rmse_list):.3f}")
    print("=" * 50)
    
    # Train final model on the entire dataset
    final_model = sm.OLS(y, X_with_const).fit()
    print("Final Model Summary (trained on the entire dataset):")
    print(final_model.summary())
    
    # Evaluate final model metrics on the entire dataset
    y_final_pred = final_model.predict(X_with_const)
    final_mae = mae(y, y_final_pred)
    final_rmse = np.sqrt(mse(y, y_final_pred))
    print()
    print("Final Model Metrics on the Entire Dataset:")
    print(f"  MAE: {final_mae:.3f}")
    print(f"  RMSE: {final_rmse:.3f}")
    
    return final_model


def plot_diagnostics(model):
    """
    Creates side-by-side QQ plot and Residuals vs. Fitted Values scatterplot
    for regression model diagnostics.
    
    Parameters:
        model: A fitted statsmodels OLS regression model.
    """
    # Get residuals and fitted values from the model
    residuals = model.resid
    fitted = model.fittedvalues
    
    # Create side-by-side subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # QQ Plot: Checks normality of the residuals
    sm.qqplot(residuals, line='s', ax=axes[0])
    axes[0].set_title("QQ Plot of Residuals")
    
    # Scatterplot: Residuals vs. Fitted Values to check for heteroskedasticity
    axes[1].scatter(fitted, residuals, alpha=0.5)
    axes[1].axhline(0, color='red', linestyle='--')
    axes[1].set_title("Residuals vs Fitted Values")
    axes[1].set_xlabel("Fitted Values")
    axes[1].set_ylabel("Residuals")
    
    plt.tight_layout()
    plt.show()