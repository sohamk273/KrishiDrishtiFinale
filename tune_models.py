"""
tune_models.py
==============
KrishiDrishti – Systematic Hyperparameter Tuning and Model Optimization Engine

Implements:
  - TimeSeriesSplit chronological cross-validation (no data leakage).
  - RandomizedSearchCV initial exploration + targeted GridSearchCV refinement.
  - Evaluation of MAE, RMSE, MAPE, and R² on CV folds and final holdout test set.
  - Scientific comparison of Original vs. Optimized configurations.
  - Persistence of trained model artifacts, search trials CSV, and JSON metadata.
"""

import os
import json
import pickle
import time
import datetime
import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import xgboost as xgb

RANDOM_STATE = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models", "optimized")

# ─────────────────────────────────────────────────────────────────────────────
# COMMODITY SPECIFICATIONS
# ─────────────────────────────────────────────────────────────────────────────
COMMODITIES = {
    "onion": {
        "name": "Onion (Lasalgaon)",
        "csv": "master_dataset.csv",
        "primary_algo": "XGBoost",
        "train_cutoff_year": 2023,
        "test_year": 2024,
        "out_dir": os.path.join(MODELS_DIR, "onion"),
    },
    "potato": {
        "name": "Potato (Achalda)",
        "csv": "master_dataset_achalda.csv",
        "primary_algo": "GradientBoosting",
        "train_cutoff_year": 2025,
        "test_year": 2026,
        "out_dir": os.path.join(MODELS_DIR, "potato"),
    },
    "turdal": {
        "name": "Tur Dal (Latur)",
        "csv": "master_dataset_latur.csv",
        "primary_algo": "GradientBoosting",
        "train_cutoff_year": 2023,
        "test_year": 2024,
        "out_dir": os.path.join(MODELS_DIR, "turdal"),
    },
}

HORIZONS = [7, 14, 30]

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING (Matching Production Repository Exactly)
# ─────────────────────────────────────────────────────────────────────────────
def build_features(df_input: pd.DataFrame, is_legacy: bool = False):
    temp = df_input.copy()
    temp['Date'] = pd.to_datetime(temp['Date'])
    temp = temp.sort_values('Date').reset_index(drop=True)

    # Impute missing environmental indicators if absent
    if 'Diesel_Price_Rs' not in temp.columns:
        temp['Diesel_Price_Rs'] = 90.0
    if 'Temperature_C' not in temp.columns:
        temp['Temperature_C'] = 25.0
    if 'Rainfall_mm' not in temp.columns:
        temp['Rainfall_mm'] = 50.0
    if 'Arrivals_Tonnes' not in temp.columns:
        temp['Arrivals_Tonnes'] = 100.0

    # Forward-fill / back-fill any numeric gaps
    num_cols = temp.select_dtypes(include=['number']).columns
    temp[num_cols] = temp[num_cols].ffill().bfill()

    if is_legacy:
        for lag in [1, 3, 7]:
            temp[f'Price_Lag_{lag}'] = temp['Modal_Price'].shift(lag)
        temp['Rolling_Mean_7d'] = temp['Modal_Price'].shift(1).rolling(window=7).mean()
        temp['Rolling_Mean_30d'] = temp['Modal_Price'].shift(1).rolling(window=30).mean()
        temp['Month'] = temp['Date'].dt.month

        features = [
            'Temperature_C', 'Rainfall_mm', 'Arrivals_Tonnes', 'Diesel_Price_Rs',
            'Modal_Price', 'Price_Lag_1', 'Price_Lag_3', 'Price_Lag_7',
            'Rolling_Mean_7d', 'Rolling_Mean_30d', 'Month'
        ]
    else:
        for lag in [1, 2, 3, 4, 5, 7, 10, 14, 21]:
            temp[f'Price_Lag_{lag}'] = temp['Modal_Price'].shift(lag)
        temp['Rolling_Mean_3d'] = temp['Modal_Price'].shift(1).rolling(window=3).mean()
        temp['Rolling_Mean_7d'] = temp['Modal_Price'].shift(1).rolling(window=7).mean()
        temp['Rolling_Mean_14d'] = temp['Modal_Price'].shift(1).rolling(window=14).mean()
        temp['Rolling_Std_7d'] = temp['Modal_Price'].shift(1).rolling(window=7).std()
        temp['Month'] = temp['Date'].dt.month
        temp['DayOfWeek'] = temp['Date'].dt.dayofweek
        temp['DayOfYear'] = temp['Date'].dt.dayofyear

        features = [
            'Temperature_C', 'Rainfall_mm', 'Arrivals_Tonnes', 'Diesel_Price_Rs',
            'Modal_Price', 'Price_Lag_1', 'Price_Lag_2', 'Price_Lag_3', 'Price_Lag_4',
            'Price_Lag_5', 'Price_Lag_7', 'Price_Lag_10', 'Price_Lag_14', 'Price_Lag_21',
            'Rolling_Mean_3d', 'Rolling_Mean_7d', 'Rolling_Mean_14d', 'Rolling_Std_7d',
            'Month', 'DayOfWeek', 'DayOfYear'
        ]

    return temp, features


def prepare_dataset(csv_file: str, horizon_days: int, is_legacy: bool):
    raw_df = pd.read_csv(csv_file)
    feat_df, features = build_features(raw_df, is_legacy=is_legacy)

    target_col = f'Target_Price_{horizon_days}d'
    feat_df[target_col] = feat_df['Modal_Price'].shift(-horizon_days)
    feat_df = feat_df.dropna().reset_index(drop=True)
    return feat_df, features, target_col


# ─────────────────────────────────────────────────────────────────────────────
# SEARCH SPACES DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
def get_search_spaces(algo: str):
    if algo == "XGBoost":
        param_dist = {
            'n_estimators': [150, 200, 250, 300, 400],
            'learning_rate': [0.015, 0.03, 0.05, 0.08, 0.1],
            'max_depth': [3, 4, 5, 6, 7],
            'min_child_weight': [1, 3, 5, 7],
            'subsample': [0.65, 0.75, 0.85, 0.95],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'gamma': [0.0, 0.1, 0.2, 0.4],
            'reg_alpha': [0.0, 0.05, 0.2, 1.0],
            'reg_lambda': [0.5, 1.0, 2.0, 5.0]
        }
        base_estimator = xgb.XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    elif algo == "GradientBoosting":
        param_dist = {
            'n_estimators': [150, 200, 300, 400],
            'learning_rate': [0.01, 0.015, 0.03, 0.05, 0.08],
            'max_depth': [3, 4, 5, 6],
            'min_samples_split': [4, 8, 12, 16],
            'min_samples_leaf': [2, 4, 6, 8],
            'subsample': [0.7, 0.8, 0.85, 0.95],
            'max_features': [None, 'sqrt', 0.8]
        }
        base_estimator = GradientBoostingRegressor(random_state=RANDOM_STATE)
    elif algo == "RandomForest":
        param_dist = {
            'n_estimators': [150, 200, 300],
            'max_depth': [8, 12, 16, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': [1.0, 'sqrt', 0.8],
            'max_samples': [None, 0.75, 0.9]
        }
        base_estimator = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    return base_estimator, param_dist


def get_original_model(commodity_key: str, horizon: int, algo: str):
    """Retrieve original production model hyperparameters."""
    if algo == "XGBoost":
        return xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    elif algo == "RandomForest":
        if commodity_key == "onion":
            return RandomForestRegressor(n_estimators=300, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=1.0, random_state=RANDOM_STATE, n_jobs=-1)
        else:
            return RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_split=2, min_samples_leaf=1, max_features=1.0, random_state=RANDOM_STATE, n_jobs=-1)
    else:  # GradientBoosting
        if commodity_key == "turdal" and horizon == 7:
            return GradientBoostingRegressor(n_estimators=400, learning_rate=0.01, max_depth=6, subsample=0.8, min_samples_split=2, min_samples_leaf=1, random_state=RANDOM_STATE)
        else:
            return GradientBoostingRegressor(n_estimators=300, learning_rate=0.015, max_depth=5, subsample=0.85, min_samples_split=10, min_samples_leaf=5, random_state=RANDOM_STATE)


# ─────────────────────────────────────────────────────────────────────────────
# TIME SERIES CROSS-VALIDATION EVALUATOR
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_ts_cv(estimator, X, y, tscv):
    fold_maes = []
    fold_rmses = []
    fold_mapes = []
    fold_r2s = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        model = estimator.__class__(**estimator.get_params())
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)

        mae = mean_absolute_error(y_val, preds)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        mape = mean_absolute_percentage_error(y_val, preds) * 100
        r2 = r2_score(y_val, preds)

        fold_maes.append(mae)
        fold_rmses.append(rmse)
        fold_mapes.append(mape)
        fold_r2s.append(r2)

    return {
        "cv_mae_mean": float(np.mean(fold_maes)),
        "cv_mae_std": float(np.std(fold_maes)),
        "cv_rmse_mean": float(np.mean(fold_rmses)),
        "cv_mape_mean": float(np.mean(fold_mapes)),
        "cv_r2_mean": float(np.mean(fold_r2s)),
        "folds": [
            {
                "fold": i + 1,
                "train_size": len(train_idx),
                "val_size": len(val_idx),
                "mae": float(fold_maes[i]),
                "rmse": float(fold_rmses[i]),
                "mape": float(fold_mapes[i]),
                "r2": float(fold_r2s[i]),
            }
            for i, (train_idx, val_idx) in enumerate(tscv.split(X))
        ]
    }


def evaluate_holdout(model, X_test, y_test):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mape = mean_absolute_percentage_error(y_test, preds) * 100
    r2 = r2_score(y_test, preds)
    return {
        "test_mae": float(mae),
        "test_rmse": float(rmse),
        "test_mape": float(mape),
        "test_r2": float(r2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEMATIC OPTIMIZATION ROUTINE
# ─────────────────────────────────────────────────────────────────────────────
def optimize_single(comm_key: str, horizon: int, algo: str, n_iter_random: int = 16):
    cfg = COMMODITIES[comm_key]
    is_legacy = (comm_key == "onion" and horizon == 7)
    df, features, target_col = prepare_dataset(cfg["csv"], horizon, is_legacy)

    # Chronological Split: Train (<= cutoff_year), Test (holdout test)
    train_mask = df['Date'].dt.year <= cfg['train_cutoff_year']
    test_mask = df['Date'].dt.year >= cfg['test_year']

    train_df = df[train_mask].reset_index(drop=True)
    test_df = df[test_mask].reset_index(drop=True)

    X_train, y_train = train_df[features], train_df[target_col]
    X_test, y_test = test_df[features], test_df[target_col]

    # TimeSeriesSplit cross-validator (3 splits)
    tscv = TimeSeriesSplit(n_splits=3)

    # 1. Evaluate Original Model
    orig_model = get_original_model(comm_key, horizon, algo)
    orig_cv_results = evaluate_ts_cv(orig_model, X_train, y_train, tscv)
    orig_model.fit(X_train, y_train)
    orig_test_results = evaluate_holdout(orig_model, X_test, y_test)
    orig_params = {k: v for k, v in orig_model.get_params().items() if k in [
        'n_estimators', 'learning_rate', 'max_depth', 'subsample', 'min_samples_split',
        'min_samples_leaf', 'max_features', 'min_child_weight', 'gamma', 'reg_alpha', 'reg_lambda'
    ] and v is not None}

    # 2. Hyperparameter Optimization: Phase 1 (RandomizedSearchCV)
    base_est, param_dist = get_search_spaces(algo)
    random_search = RandomizedSearchCV(
        estimator=base_est,
        param_distributions=param_dist,
        n_iter=n_iter_random,
        scoring='neg_mean_absolute_error',
        cv=tscv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        return_train_score=False
    )
    random_search.fit(X_train, y_train)

    best_random_params = random_search.best_params_

    # 3. Phase 2: Targeted Grid Search refinement around best parameters
    # Construct a compact 2x2 grid around the top 2 parameters
    grid_params = {}
    if algo in ["XGBoost", "GradientBoosting"]:
        lr = best_random_params.get('learning_rate', 0.05)
        n_est = best_random_params.get('n_estimators', 300)
        grid_params['learning_rate'] = sorted(list(set([max(0.01, round(lr * 0.8, 4)), lr, round(lr * 1.2, 4)])))
        grid_params['n_estimators'] = sorted(list(set([max(100, n_est - 50), n_est, n_est + 50])))
        for k, v in best_random_params.items():
            if k not in ['learning_rate', 'n_estimators']:
                grid_params[k] = [v]
    else:  # RandomForest
        n_est = best_random_params.get('n_estimators', 300)
        grid_params['n_estimators'] = sorted(list(set([max(100, n_est - 50), n_est, n_est + 50])))
        for k, v in best_random_params.items():
            if k not in ['n_estimators']:
                grid_params[k] = [v]

    grid_search = GridSearchCV(
        estimator=base_est,
        param_grid=grid_params,
        scoring='neg_mean_absolute_error',
        cv=tscv,
        n_jobs=-1,
        return_train_score=False
    )
    grid_search.fit(X_train, y_train)

    # Determine absolute best estimator
    if grid_search.best_score_ > random_search.best_score_:
        best_est = grid_search.best_estimator_
        best_params = grid_search.best_params_
    else:
        best_est = random_search.best_estimator_
        best_params = random_search.best_params_

    # 4. Evaluate Optimized Model
    opt_cv_results = evaluate_ts_cv(best_est, X_train, y_train, tscv)
    best_est.fit(X_train, y_train)
    opt_test_results = evaluate_holdout(best_est, X_test, y_test)

    # 5. Extract Search Trials for Analysis
    cv_df_random = pd.DataFrame(random_search.cv_results_)
    cv_df_grid = pd.DataFrame(grid_search.cv_results_)
    all_trials = pd.concat([cv_df_random, cv_df_grid], ignore_index=True)
    all_trials['mean_test_mae'] = -all_trials['mean_test_score']
    all_trials['rank'] = all_trials['mean_test_mae'].rank().astype(int)

    # Select clean columns for trial export
    param_col_keys = [c for c in all_trials.columns if c.startswith('param_')]
    clean_cols = ['rank', 'mean_test_mae', 'std_test_score'] + param_col_keys
    export_trials = all_trials[clean_cols].sort_values('rank').reset_index(drop=True)
    export_trials.columns = [c.replace('param_', '') for c in export_trials.columns]

    # Calculate Improvement Percentages
    mae_imp = ((orig_test_results['test_mae'] - opt_test_results['test_mae']) / orig_test_results['test_mae']) * 100
    rmse_imp = ((orig_test_results['test_rmse'] - opt_test_results['test_rmse']) / orig_test_results['test_rmse']) * 100
    mape_imp = ((orig_test_results['test_mape'] - opt_test_results['test_mape']) / orig_test_results['test_mape']) * 100
    cv_mae_imp = ((orig_cv_results['cv_mae_mean'] - opt_cv_results['cv_mae_mean']) / orig_cv_results['cv_mae_mean']) * 100

    # 6. Save Model Artifact
    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    slug = f"{comm_key}_{algo.lower()}_{horizon}d"
    model_path = os.path.join(out_dir, f"{slug}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(best_est, f)

    # 7. Save Trials CSV
    trials_csv_path = os.path.join(out_dir, f"search_results_{algo.lower()}_{horizon}d.csv")
    export_trials.to_csv(trials_csv_path, index=False)

    summary_entry = {
        "commodity": cfg["name"],
        "commodity_key": comm_key,
        "horizon_days": horizon,
        "algorithm": algo,
        "model_artifact": os.path.relpath(model_path, BASE_DIR),
        "trials_csv": os.path.relpath(trials_csv_path, BASE_DIR),
        "train_samples": len(X_train),
        "holdout_samples": len(X_test),
        "features_count": len(features),
        "features": features,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "original": {
            "hyperparameters": orig_params,
            "cv_metrics": orig_cv_results,
            "holdout_metrics": orig_test_results,
        },
        "optimized": {
            "hyperparameters": best_params,
            "cv_metrics": opt_cv_results,
            "holdout_metrics": opt_test_results,
        },
        "improvements": {
            "cv_mae_pct": round(float(cv_mae_imp), 2),
            "holdout_mae_pct": round(float(mae_imp), 2),
            "holdout_rmse_pct": round(float(rmse_imp), 2),
            "holdout_mape_pct": round(float(mape_imp), 2),
        }
    }

    print(f"[{cfg['name']} | {horizon}d | {algo}]")
    print(f"  Orig Test MAE: Rs. {orig_test_results['test_mae']:.2f} -> Opt Test MAE: Rs. {opt_test_results['test_mae']:.2f} ({mae_imp:+.2f}%)")
    print(f"  Orig CV MAE:   Rs. {orig_cv_results['cv_mae_mean']:.2f} -> Opt CV MAE:   Rs. {opt_cv_results['cv_mae_mean']:.2f} ({cv_mae_imp:+.2f}%)")

    return summary_entry


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION ROUTINE
# ─────────────────────────────────────────────────────────────────────────────
def run_all_optimizations():
    print("=" * 70)
    print("KrishiDrishti – Systematic Model Optimization & Hyperparameter Tuning")
    print("=" * 70)

    start_time = time.time()
    master_summary = {}

    for comm_key, cfg in COMMODITIES.items():
        master_summary[comm_key] = {}
        primary_algo = cfg["primary_algo"]
        algos = [primary_algo, "RandomForest"]

        for horizon in HORIZONS:
            master_summary[comm_key][f"{horizon}d"] = {}
            for algo in algos:
                entry = optimize_single(comm_key, horizon, algo)
                master_summary[comm_key][f"{horizon}d"][algo] = entry

        # Save commodity-level summary
        comm_json_path = os.path.join(cfg["out_dir"], "metrics_summary.json")
        with open(comm_json_path, 'w', encoding='utf-8') as f:
            json.dump(master_summary[comm_key], f, indent=2)
        print(f"Saved {comm_json_path}")

    # Save overall master summary
    master_json_path = os.path.join(MODELS_DIR, "master_optimization_summary.json")
    with open(master_json_path, 'w', encoding='utf-8') as f:
        json.dump(master_summary, f, indent=2)

    elapsed = time.time() - start_time
    print("=" * 70)
    print(f"[OK] All 18 model optimizations completed successfully in {elapsed:.1f} seconds!")
    print(f"Master metadata saved to: {master_json_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_all_optimizations()
