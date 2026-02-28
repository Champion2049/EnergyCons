import os
import time
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import clone
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. CONFIGURATION
# ==========================================
ORIGINAL_DATA_PATH = 'Wavelet-LWT-Rtamanyu/maams_preprocessed_yield_weather.csv'
LWT_L1_DATA = 'Wavelet-LWT-Rtamanyu/lwt_level_1_preprocessed_yield_weather.csv'
LWT_L2_DATA = 'Wavelet-LWT-Rtamanyu/lwt_level_2_preprocessed_yield_weather.csv'
OUTPUT_DIR = 'Wavelet-LWT-Rtamanyu/Output'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Helper to create mock LWT datasets if they don't exist yet
if os.path.exists(ORIGINAL_DATA_PATH):
    df_mock = pd.read_csv(ORIGINAL_DATA_PATH)
    if not os.path.exists(LWT_L1_DATA):
        print(f"Generating mock {LWT_L1_DATA} for demonstration...")
        df_mock.to_csv(LWT_L1_DATA, index=False)
    if not os.path.exists(LWT_L2_DATA):
        print(f"Generating mock {LWT_L2_DATA} for demonstration...")
        df_mock.to_csv(LWT_L2_DATA, index=False)

# ==========================================
# 2. DATA PREPARATION
# ==========================================
def prepare_data(filepath):
    """Loads, time-splits, and scales data (Required for SVR)."""
    df = pd.read_csv(filepath)
    train = df[df['Year'] < 2015]
    test = df[df['Year'] >= 2015]
    
    X_train = train.drop(columns=['Year', 'Yield'])
    y_train = train['Yield']
    X_test = test.drop(columns=['Year', 'Yield'])
    y_test = test['Yield']
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

# ==========================================
# 3. HYPERPARAMETER TUNING
# ==========================================
def tune_models(X_train, y_train):
    print("\n--- Starting Hyperparameter Tuning on Original Data ---")
    
    # 1. Tune XGBoost
    print("Tuning XGBoost...")
    xgb_grid = {'n_estimators': [50, 100, 150], 'max_depth': [3, 5, 7], 'learning_rate': [0.05, 0.1, 0.2]}
    xgb_search = GridSearchCV(xgb.XGBRegressor(random_state=42, objective='reg:squarederror'), 
                              xgb_grid, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1)
    xgb_search.fit(X_train, y_train)
    best_xgb = xgb_search.best_estimator_
    
    # 2. Tune Random Forest
    print("Tuning Random Forest...")
    rf_grid = {'n_estimators': [50, 100, 150], 'max_depth': [None, 5, 10]}
    rf_search = GridSearchCV(RandomForestRegressor(random_state=42), 
                             rf_grid, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1)
    rf_search.fit(X_train, y_train)
    best_rf = rf_search.best_estimator_
    
    # 3. Tune SVR
    print("Tuning SVR...")
    svr_grid = {'C': [0.1, 1, 10, 100], 'epsilon': [0.01, 0.1, 1], 'kernel': ['rbf', 'linear']}
    svr_search = GridSearchCV(SVR(), svr_grid, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1)
    svr_search.fit(X_train, y_train)
    best_svr = svr_search.best_estimator_
    
    # 4. Create Stacking Regressor
    print("Building Stacking Regressor...")
    estimators = [('xgb', best_xgb), ('rf', best_rf), ('svr', best_svr)]
    best_stack = StackingRegressor(estimators=estimators, final_estimator=Ridge())
    
    print("Tuning Complete!")
    return {'XGBoost': best_xgb, 'RandomForest': best_rf, 'SVR': best_svr, 'Stacking': best_stack}

# ==========================================
# 4. TRAINING & EVALUATION
# ==========================================
def evaluate_models_on_data(models_dict, filepath, dataset_name):
    print(f"\n--- Evaluating on {dataset_name} ---")
    X_train, X_test, y_train, y_test = prepare_data(filepath)
    
    metrics = []
    trained_models = {}
    
    for model_name, model in models_dict.items():
        cloned_model = clone(model)
        
        start_time = time.time()
        cloned_model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        preds = cloned_model.predict(X_test)
        
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        
        print(f"[{model_name}] RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.4f} | Time: {train_time:.2f}s")
        
        # Save model for hypothesis testing later
        trained_models[model_name] = cloned_model
        
        metrics.append({
            'Dataset': dataset_name,
            'Model': model_name,
            'MAE': mae,
            'RMSE': rmse,
            'R2_Score': r2,
            'Train_Time_sec': train_time
        })
        
    return metrics, trained_models

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    all_metrics = []
    
    # Step 1: Tune models on Original Data
    X_train_orig, _, y_train_orig, _ = prepare_data(ORIGINAL_DATA_PATH)
    tuned_models = tune_models(X_train_orig, y_train_orig)
    
    # Step 2: Evaluate and Save trained models
    datasets = [
        (ORIGINAL_DATA_PATH, 'Original'),
        (LWT_L1_DATA, 'LWT_L1'),
        (LWT_L2_DATA, 'LWT_L2')
    ]
    
    for filepath, name in datasets:
        if os.path.exists(filepath):
            metrics, trained = evaluate_models_on_data(tuned_models, filepath, name)
            all_metrics.extend(metrics)
            
            # Save trained models for Hypothesis testing script
            for model_name, fitted_model in trained.items():
                model_path = os.path.join(OUTPUT_DIR, f"{model_name}_{name}.joblib")
                joblib.dump(fitted_model, model_path)
        else:
            print(f"Skipping {name}, file '{filepath}' not found.")
            
    # Step 3: Save CSV
    results_df = pd.DataFrame(all_metrics)
    csv_out_path = os.path.join(OUTPUT_DIR, 'advanced_ml_metrics.csv')
    results_df.to_csv(csv_out_path, index=False)
    print(f"\nSaved metrics to: {csv_out_path}")
    
    # Step 4: Plot
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(12, 16))
    fig.suptitle('Model Performance Across Datasets', fontsize=16, fontweight='bold')
    
    def add_labels(ax):
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='bottom', xytext=(0, 3), textcoords='offset points', fontsize=9)

    sns.barplot(data=results_df, x='Model', y='RMSE', hue='Dataset', ax=axes[0], palette='viridis')
    axes[0].set_title('Root Mean Squared Error (Lower is Better)')
    add_labels(axes[0])
    
    sns.barplot(data=results_df, x='Model', y='MAE', hue='Dataset', ax=axes[1], palette='viridis')
    axes[1].set_title('Mean Absolute Error (Lower is Better)')
    add_labels(axes[1])
    
    sns.barplot(data=results_df, x='Model', y='R2_Score', hue='Dataset', ax=axes[2], palette='viridis')
    axes[2].set_title('R² Score (Higher is Better)')
    axes[2].set_ylim(-0.2, 1.0)
    add_labels(axes[2])
    
    plt.tight_layout(rect=(0, 0.03, 1, 0.96))
    plot_path = os.path.join(OUTPUT_DIR, 'advanced_ml_comparison.png')
    plt.savefig(plot_path, dpi=300)
    print(f"Saved comparison charts to: {plot_path}")