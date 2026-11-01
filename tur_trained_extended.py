import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error
import pickle

print("=== Starting Tur Dal Multi-Horizon Training Pipeline (14d & 30d) ===")

# 1. Load your cleaned, non-redundant Tur Dal dataset
DATASET_FILENAME = 'master_dataset_latur.csv'

try:
    df = pd.read_csv(DATASET_FILENAME)
except FileNotFoundError:
    print(f"Error: Could not find '{DATASET_FILENAME}'. Please ensure your Tur Dal CSV is in the folder.")
    exit()

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# 2. Elite Feature Engineering (Lags and Rolling Statistics)
for lag in [1, 2, 3, 4, 5, 7, 10, 14, 21]:
    df[f'Price_Lag_{lag}'] = df['Modal_Price'].shift(lag)

df['Rolling_Mean_3d'] = df['Modal_Price'].shift(1).rolling(window=3).mean()
df['Rolling_Mean_7d'] = df['Modal_Price'].shift(1).rolling(window=7).mean()
df['Rolling_Mean_14d'] = df['Modal_Price'].shift(1).rolling(window=14).mean()
df['Rolling_Std_7d'] = df['Modal_Price'].shift(1).rolling(window=7).std()

df['Month'] = df['Date'].dt.month
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['DayOfYear'] = df['Date'].dt.dayofyear

features_elite = [
    'Temperature_C', 'Rainfall_mm', 'Arrivals_Tonnes', 'Diesel_Price_Rs', 
    'Modal_Price', 'Price_Lag_1', 'Price_Lag_2', 'Price_Lag_3', 'Price_Lag_4',
    'Price_Lag_5', 'Price_Lag_7', 'Price_Lag_10', 'Price_Lag_14', 'Price_Lag_21', 
    'Rolling_Mean_3d', 'Rolling_Mean_7d', 'Rolling_Mean_14d', 'Rolling_Std_7d', 
    'Month', 'DayOfWeek', 'DayOfYear'
]

# 3. Train and Evaluate for 14-Day and 30-Day Horizons
horizons = {
    14: 'turdal_model_14d.pkl',
    30: 'turdal_model_30d.pkl'
}

for days, filename in horizons.items():
    print(f"\n--- Training Tur Dal {days}-Day Horizon Model ---")
    
    target_col = f'Target_Price_{days}d'
    df_h = df.copy()
    df_h[target_col] = df_h['Modal_Price'].shift(-days)
    df_h.dropna(inplace=True)
    
    # Time-series split (Training on <= 2023, Testing on 2024+)
    train_df = df_h[df_h['Date'].dt.year <= 2023]
    test_df = df_h[df_h['Date'].dt.year >= 2024]
    
    X_train = train_df[features_elite]
    y_train = train_df[target_col]
    
    # Regularized Gradient Boosting Regressor (Anti-Overfitting)
    model = GradientBoostingRegressor(
        n_estimators=300, 
        learning_rate=0.015, 
        max_depth=5, 
        subsample=0.85, 
        min_samples_split=10, 
        min_samples_leaf=5, 
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate Test Accuracy if test set exists
    if len(test_df) > 0:
        X_test = test_df[features_elite]
        y_test = test_df[target_col]
        predictions = model.predict(X_test)
        
        mape = mean_absolute_percentage_error(y_test, predictions)
        accuracy = max(0, (1 - mape) * 100)
        print(f"[Tur Dal {days}-Day Model] Test MAPE: {mape:.4f} | Accuracy: {accuracy:.2f}%")
    
    # Save the trained model file
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Successfully saved model to '{filename}'")

print("\n=== All Tur Dal 14d & 30d Models Trained & Saved Successfully! ===")
