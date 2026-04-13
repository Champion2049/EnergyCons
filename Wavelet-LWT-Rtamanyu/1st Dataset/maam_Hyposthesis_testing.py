import os
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
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
results_file = open(os.path.join(OUTPUT_DIR, 'hypothesis_test_results.txt'), 'w')

def log_result(text):
    print(text)
    results_file.write(text + "\n")

def get_test_data(filepath):
    df = pd.read_csv(filepath)
    test = df[df['Year'] >= 2015]
    X_test = test.drop(columns=['Year', 'Yield'])
    y_test = test['Yield']
    
    # We must scale X_test just like we did in training
    train = df[df['Year'] < 2015]
    X_train = train.drop(columns=['Year', 'Yield'])
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_test_scaled, y_test.values

# ==========================================
# APPROACH 1: PREDICTION-LEVEL ERRORS
# ==========================================
def test_approach_1_predictions():
    log_result("\n" + "="*60)
    log_result("APPROACH 1: PREDICTION-LEVEL ERRORS (WILCOXON TEST)")
    log_result("="*60)
    
    datasets = {
        'Original': ORIGINAL_DATA_PATH,
        'LWT_L1': LWT_L1_DATA,
        'LWT_L2': LWT_L2_DATA
    }
    models = ['XGBoost', 'RandomForest', 'SVR', 'Stacking']
    
    plot_data = [] # To store data for our boxplot
    
    for model_name in models:
        log_result(f"\n--- Testing {model_name} ---")
        try:
            # Load Datasets
            X_orig, y_true = get_test_data(datasets['Original'])
            X_l1, _ = get_test_data(datasets['LWT_L1'])
            X_l2, _ = get_test_data(datasets['LWT_L2'])
            
            # Load Models
            model_orig = joblib.load(os.path.join(OUTPUT_DIR, f"{model_name}_Original.joblib"))
            model_l1 = joblib.load(os.path.join(OUTPUT_DIR, f"{model_name}_LWT_L1.joblib"))
            model_l2 = joblib.load(os.path.join(OUTPUT_DIR, f"{model_name}_LWT_L2.joblib"))
            
            # Generate Predictions and absolute errors
            err_orig = np.abs(y_true - model_orig.predict(X_orig))
            err_l1 = np.abs(y_true - model_l1.predict(X_l1))
            err_l2 = np.abs(y_true - model_l2.predict(X_l2))
            
            # Append data for plotting
            for e in err_orig: plot_data.append({'Model': model_name, 'Dataset': 'Original', 'Absolute Error': e})
            for e in err_l1: plot_data.append({'Model': model_name, 'Dataset': 'LWT Level 1', 'Absolute Error': e})
            for e in err_l2: plot_data.append({'Model': model_name, 'Dataset': 'LWT Level 2', 'Absolute Error': e})
            
            # L1 vs Original Wilcoxon Test
            stat_l1, p_l1 = stats.wilcoxon(err_orig, err_l1, alternative='greater')
            log_result(f"Orig vs LWT_L1 -> Orig MAE: {np.mean(err_orig):.2f} | L1 MAE: {np.mean(err_l1):.2f} | P-Value: {p_l1:.4f}")
            if p_l1 < 0.05: log_result("  * LWT Level 1 is significantly better.") # type: ignore
            
            # L2 vs Original Wilcoxon Test
            stat_l2, p_l2 = stats.wilcoxon(err_orig, err_l2, alternative='greater')
            log_result(f"Orig vs LWT_L2 -> Orig MAE: {np.mean(err_orig):.2f} | L2 MAE: {np.mean(err_l2):.2f} | P-Value: {p_l2:.4f}")
            if p_l2 < 0.05: log_result("  * LWT Level 2 is significantly better.") # type: ignore
            
        except Exception as e:
            log_result(f"Skipping {model_name} due to error or missing files: {str(e)}")

    # -- PICTORIAL REPRESENTATION: Grouped Boxplot --
    if plot_data:
        plt.figure(figsize=(14, 7))
        sns.boxplot(data=pd.DataFrame(plot_data), x='Model', y='Absolute Error', hue='Dataset', palette='Set2')
        plt.title('Distribution of Prediction Errors by Model and Dataset (Test Set)', fontsize=15, fontweight='bold')
        plt.ylabel('Absolute Error (Yield Units)', fontsize=12)
        plt.xlabel('Machine Learning Model', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plot_path = os.path.join(OUTPUT_DIR, 'approach1_error_distributions.png')
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()
        log_result(f"\n-> Saved Error Distribution Boxplot to: {plot_path}")


# ==========================================
# APPROACH 2: CROSS-VALIDATION FOLD SCORES
# ==========================================
def perform_cv_for_dataset(filepath, model_base):
    df = pd.read_csv(filepath)
    test_years = sorted(df['Year'].unique())[-5:] # Last 5 years
    cv_scores = []
    
    for target_year in test_years:
        train = df[df['Year'] < target_year]
        test = df[df['Year'] == target_year]
        
        X_train, y_train = train.drop(columns=['Year', 'Yield']), train['Yield']
        X_test, y_test = test.drop(columns=['Year', 'Yield']), test['Yield']
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        # Clone to reset weights but keep hyperparams
        from sklearn.base import clone
        model = clone(model_base)
        model.fit(X_train_s, y_train)
        
        rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test_s)))
        cv_scores.append(rmse)
        
    return test_years, cv_scores

def test_approach_2_cv():
    log_result("\n" + "="*60)
    log_result("APPROACH 2: TIME-SERIES CROSS-VALIDATION (PAIRED T-TEST)")
    log_result("="*60)
    
    models = ['XGBoost', 'RandomForest', 'SVR', 'Stacking']
    
    # Setup a 2x2 grid for plotting the 4 models
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Time-Series CV: RMSE Across Validation Years', fontsize=18, fontweight='bold')
    axes = axes.flatten()
    
    for idx, model_name in enumerate(models):
        log_result(f"\n--- CV Testing {model_name} ---")
        try:
            # Load template model
            model_base = joblib.load(os.path.join(OUTPUT_DIR, f"{model_name}_Original.joblib"))
            
            years, scores_orig = perform_cv_for_dataset(ORIGINAL_DATA_PATH, model_base)
            _, scores_l1 = perform_cv_for_dataset(LWT_L1_DATA, model_base)
            _, scores_l2 = perform_cv_for_dataset(LWT_L2_DATA, model_base)
            
            # T-test computations
            stat_l1, p_l1 = stats.ttest_rel(scores_orig, scores_l1, alternative='greater')
            log_result(f"Orig vs L1 -> Orig Mean CV: {np.mean(scores_orig):.2f} | L1 Mean CV: {np.mean(scores_l1):.2f} | P-Value: {p_l1:.4f}")
            if p_l1 < 0.05: log_result("  * LWT Level 1 consistently reduces RMSE.")
            
            stat_l2, p_l2 = stats.ttest_rel(scores_orig, scores_l2, alternative='greater')
            log_result(f"Orig vs L2 -> Orig Mean CV: {np.mean(scores_orig):.2f} | L2 Mean CV: {np.mean(scores_l2):.2f} | P-Value: {p_l2:.4f}")
            if p_l2 < 0.05: log_result("  * LWT Level 2 consistently reduces RMSE.")
            
            # -- PICTORIAL REPRESENTATION: Line Chart for this model --
            ax = axes[idx]
            ax.plot(years, scores_orig, marker='o', label='Original Data', color='#e74c3c', linewidth=2.5, markersize=8)
            ax.plot(years, scores_l1, marker='s', label='LWT Level 1', color='#3498db', linewidth=2.5, markersize=8)
            ax.plot(years, scores_l2, marker='^', label='LWT Level 2', color='#2ecc71', linewidth=2.5, markersize=8)
            
            ax.set_title(f'{model_name} Performance over Time', fontsize=14)
            ax.set_xlabel('Validation Year', fontsize=11)
            ax.set_ylabel('RMSE', fontsize=11)
            ax.set_xticks(years)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend()
            
        except Exception as e:
            log_result(f"Skipping {model_name} due to error: {str(e)}")
            axes[idx].set_title(f'{model_name} (Data Unavailable)')
            axes[idx].axis('off')

    # Save the 2x2 grid chart
    plt.tight_layout(rect=(0, 0.03, 1, 0.96))
    plot_path = os.path.join(OUTPUT_DIR, 'approach2_cv_scores_grid.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    log_result(f"\n-> Saved CV Line Charts Grid to: {plot_path}")

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(os.path.join(OUTPUT_DIR, 'XGBoost_Original.joblib')):
        print("Please run 'advanced_ml_comparison.py' first so the models are trained and saved!")
    else:
        test_approach_1_predictions()
        test_approach_2_cv()
        
        results_file.close()
        print("\nAll Hypothesis testing complete! Check the Output folder for the text log and the generated PNG plots.")