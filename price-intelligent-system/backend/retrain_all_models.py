"""
retrain_all_models.py
Retrains all commodity models (7d, 14d, 30d horizons) using the current
environment's xgboost/sklearn versions and saves them with pickle.
Run from: backend/ directory
  python retrain_all_models.py
"""
import os, pickle, warnings
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
import xgboost as xgb

warnings.filterwarnings("ignore")

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ─── FEATURE ENGINEERING ──────────────────────────────────────────────────────
def make_onion_features(df):
    df = df.copy()
    for lag in [1, 3, 7]:
        df[f'Price_Lag_{lag}'] = df['Modal_Price'].shift(lag)
    df['Rolling_Mean_7d']  = df['Modal_Price'].shift(1).rolling(7).mean()
    df['Rolling_Mean_30d'] = df['Modal_Price'].shift(1).rolling(30).mean()
    df['Month'] = df['Date'].dt.month
    return df

def make_advanced_features(df):
    df = df.copy()
    for lag in [1, 2, 3, 4, 5, 7, 10, 14, 21]:
        df[f'Price_Lag_{lag}'] = df['Modal_Price'].shift(lag)
    df['Rolling_Mean_3d']  = df['Modal_Price'].shift(1).rolling(3).mean()
    df['Rolling_Mean_7d']  = df['Modal_Price'].shift(1).rolling(7).mean()
    df['Rolling_Mean_14d'] = df['Modal_Price'].shift(1).rolling(14).mean()
    df['Rolling_Std_7d']   = df['Modal_Price'].shift(1).rolling(7).std()
    df['Month']     = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['DayOfYear'] = df['Date'].dt.dayofyear
    return df

def make_turdal_features(df):
    df = df.copy()
    agg_funcs = {
        'Temperature_C':   'first', 'Rainfall_mm': 'first',
        'Arrivals_Tonnes': 'sum',   'Min_Price':   'mean',
        'Max_Price':       'mean',  'Modal_Price': 'mean',
        'Diesel_Price_Rs': 'first',
    }
    if 'Min_Price' in df.columns:
        df = df.groupby('Date').agg(agg_funcs).reset_index()
    df = df.sort_values('Date').ffill().bfill()
    return make_advanced_features(df)

ONION_FEATURES = [
    'Temperature_C', 'Rainfall_mm', 'Arrivals_Tonnes', 'Diesel_Price_Rs',
    'Modal_Price', 'Price_Lag_1', 'Price_Lag_3', 'Price_Lag_7',
    'Rolling_Mean_7d', 'Rolling_Mean_30d', 'Month'
]

ADV_FEATURES = [
    'Temperature_C', 'Rainfall_mm', 'Arrivals_Tonnes', 'Diesel_Price_Rs',
    'Modal_Price', 'Price_Lag_1', 'Price_Lag_2', 'Price_Lag_3', 'Price_Lag_4',
    'Price_Lag_5', 'Price_Lag_7', 'Price_Lag_10', 'Price_Lag_14', 'Price_Lag_21',
    'Rolling_Mean_3d', 'Rolling_Mean_7d', 'Rolling_Mean_14d', 'Rolling_Std_7d',
    'Month', 'DayOfWeek', 'DayOfYear'
]

def save_model(model, path):
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"  [SAVED] {path}")

def evaluate(model, X_test, y_test, horizon, commodity):
    preds = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, preds) * 100
    mae  = mean_absolute_error(y_test, preds)
    print(f"  [OK] {commodity} {horizon}d | MAE: Rs.{mae:.0f} | MAPE: {mape:.2f}%")
    return round(mape, 2)


# ─── ONION ────────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("ONION — Lasalgaon (XGBoost, horizons: 7, 14, 30)")
print("="*55)
df_raw = pd.read_csv('data/master_dataset.csv')
df_raw['Date'] = pd.to_datetime(df_raw['Date'])
df_raw = df_raw.sort_values('Date')
df_o = make_onion_features(df_raw)

for horizon in [7, 14, 30]:
    df = df_o.copy()
    df[f'Target'] = df['Modal_Price'].shift(-horizon)
    df.dropna(inplace=True)
    train = df[df['Date'].dt.year <= 2023]
    test  = df[df['Date'].dt.year == 2024]
    if len(test) == 0:
        test = df.tail(60)
        train = df.iloc[:-60]
    X_tr, y_tr = train[ONION_FEATURES], train['Target']
    X_te, y_te = test[ONION_FEATURES],  test['Target']
    model = xgb.XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        verbosity=0, tree_method='hist'
    )
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    evaluate(model, X_te, y_te, horizon, "Onion")
    suffix = {7: '', 14: '_14d', 30: '_30d'}[horizon]
    save_model(model, f"models/xgboost_model{suffix}.pkl")


# ─── POTATO ───────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("POTATO — Achalda (GradientBoosting, horizons: 7, 14, 30)")
print("="*55)
df_raw = pd.read_csv('data/master_dataset_achalda.csv')
df_raw['Date'] = pd.to_datetime(df_raw['Date'])
df_raw = df_raw.sort_values('Date')
df_p = make_advanced_features(df_raw)

for horizon in [7, 14, 30]:
    df = df_p.copy()
    df['Target'] = df['Modal_Price'].shift(-horizon)
    df.dropna(inplace=True)
    train = df[df['Date'].dt.year <= 2025]
    test  = df[df['Date'].dt.year == 2026]
    if len(test) < 10:
        test  = df.tail(90)
        train = df.iloc[:-90]
    X_tr, y_tr = train[ADV_FEATURES], train['Target']
    X_te, y_te = test[ADV_FEATURES],  test['Target']
    model = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.015, max_depth=5,
        subsample=0.85, min_samples_split=10, min_samples_leaf=5,
        random_state=42
    )
    model.fit(X_tr, y_tr)
    evaluate(model, X_te, y_te, horizon, "Potato")
    suffix = {7: '', 14: '_14d', 30: '_30d'}[horizon]
    save_model(model, f"models/potato_model{suffix}.pkl")


# ─── TUR DAL ──────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("TUR DAL — Latur (GradientBoosting, horizons: 7, 14, 30)")
print("="*55)
df_raw = pd.read_csv('data/master_dataset_latur.csv')
df_raw['Date'] = pd.to_datetime(df_raw['Date'])
df_raw = df_raw.sort_values('Date')
df_t = make_turdal_features(df_raw)

for horizon in [7, 14, 30]:
    df = df_t.copy()
    df['Target'] = df['Modal_Price'].shift(-horizon)
    df.dropna(inplace=True)
    train = df[df['Date'].dt.year <= 2023]
    test  = df[df['Date'].dt.year == 2024]
    if len(test) < 10:
        test  = df.tail(60)
        train = df.iloc[:-60]
    X_tr, y_tr = train[ADV_FEATURES], train['Target']
    X_te, y_te = test[ADV_FEATURES],  test['Target']
    model = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.015, max_depth=5,
        subsample=0.85, min_samples_split=10, min_samples_leaf=5,
        random_state=42
    )
    model.fit(X_tr, y_tr)
    evaluate(model, X_te, y_te, horizon, "Tur Dal")
    suffix = {7: '', 14: '_14d', 30: '_30d'}[horizon]
    save_model(model, f"models/turdal_model{suffix}.pkl")


# ─── VERIFY ALL MODELS LOADABLE ───────────────────────────────────────────────
print("\n" + "="*55)
print("VERIFICATION — Testing all models load & predict correctly")
print("="*55)

test_cases = [
    ("onion",  "xgboost_model",  "data/master_dataset.csv",        make_onion_features,    ONION_FEATURES),
    ("potato", "potato_model",   "data/master_dataset_achalda.csv", make_advanced_features, ADV_FEATURES),
    ("turdal", "turdal_model",   "data/master_dataset_latur.csv",   make_turdal_features,   ADV_FEATURES),
]

for comm, model_prefix, data_path, feat_fn, features in test_cases:
    df_v = pd.read_csv(data_path)
    df_v['Date'] = pd.to_datetime(df_v['Date'])
    df_v = df_v.sort_values('Date')
    df_v = feat_fn(df_v)
    df_clean = df_v.dropna(subset=features)
    last = df_clean.iloc[-1]
    X = last[features].to_frame().T.astype(float)  # CRITICAL: cast to float
    for suffix, label in [('', '7d'), ('_14d', '14d'), ('_30d', '30d')]:
        path = f"models/{model_prefix}{suffix}.pkl"
        with open(path, 'rb') as f:
            m = pickle.load(f)
        pred = m.predict(X)[0]
        print(f"  [PASS] {comm:8s} {label}: Current={last['Modal_Price']:.0f} -> Pred={pred:.0f}")

print("\n[DONE] All models retrained and verified successfully!")
