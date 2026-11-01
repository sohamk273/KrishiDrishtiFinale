import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

print("=== Training Random Forest Regressor for Potato (Achalda) 7-Day Forecast ===")
print("Loading Master Dataset...")
df = pd.read_csv('master_dataset_achalda.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# --- 1. Feature Engineering (Elite Multi-Horizon Features) ---
print("Generating economic indicators (Lags, Rolling Averages, and Temporal Features)...")
for lag in [1, 2, 3, 4, 5, 7, 10, 14, 21]:
    df[f'Price_Lag_{lag}'] = df['Modal_Price'].shift(lag)

df['Rolling_Mean_3d'] = df['Modal_Price'].shift(1).rolling(window=3).mean()
df['Rolling_Mean_7d'] = df['Modal_Price'].shift(1).rolling(window=7).mean()
df['Rolling_Mean_14d'] = df['Modal_Price'].shift(1).rolling(window=14).mean()
df['Rolling_Std_7d'] = df['Modal_Price'].shift(1).rolling(window=7).std()

df['Month'] = df['Date'].dt.month
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['DayOfYear'] = df['Date'].dt.dayofyear

# Safely handle potential missing columns in the dataset
if 'Diesel_Price_Rs' not in df.columns: df['Diesel_Price_Rs'] = 90.0
if 'Temperature_C' not in df.columns: df['Temperature_C'] = 25.0
if 'Rainfall_mm' not in df.columns: df['Rainfall_mm'] = 50.0
if 'Arrivals_Tonnes' not in df.columns: df['Arrivals_Tonnes'] = 100.0

# --- 2. Define the Target (7-day forecast horizon) ---
df['Target_Price_7d'] = df['Modal_Price'].shift(-7)
df.dropna(inplace=True)

# --- 3. Chronological Time-Series Split (80/20 Percentage Split) ---
print("Splitting Data: 80% Train vs 20% Test holdout...")

# Calculate the exact index for an 80% train / 20% test split dynamically
split_idx = int(len(df) * 0.8)

train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

features = [
    'Temperature_C', 'Rainfall_mm', 'Arrivals_Tonnes', 'Diesel_Price_Rs', 
    'Modal_Price', 'Price_Lag_1', 'Price_Lag_2', 'Price_Lag_3', 'Price_Lag_4',
    'Price_Lag_5', 'Price_Lag_7', 'Price_Lag_10', 'Price_Lag_14', 'Price_Lag_21', 
    'Rolling_Mean_3d', 'Rolling_Mean_7d', 'Rolling_Mean_14d', 'Rolling_Std_7d', 
    'Month', 'DayOfWeek', 'DayOfYear'
]

X_train, y_train = train_df[features], train_df['Target_Price_7d']
X_test, y_test = test_df[features], test_df['Target_Price_7d']

# --- 4. Train the Random Forest Regressor ---
print("Training RandomForestRegressor (n_estimators=300, random_state=42, n_jobs=-1)...")
rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# --- 5. Evaluate Random Forest ---
rf_preds = rf_model.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_preds)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
rf_r2 = r2_score(y_test, rf_preds)
rf_mape = mean_absolute_percentage_error(y_test, rf_preds)

print("\n=== RANDOM FOREST MODEL PERFORMANCE (Unseen Test Split) ===")
print(f"Mean Absolute Error (MAE): Rs. {rf_mae:.2f} per quintal")
print(f"Root Mean Squared Error (RMSE): Rs. {rf_rmse:.2f} per quintal")
print(f"R-squared (R²): {rf_r2:.4f}")
print(f"Mean Absolute Percentage Error (MAPE): {rf_mape:.2%}")
print("===========================================================")

# --- 6. Save Random Forest Model ---
output_filename = 'potato_rf_model.pkl'
with open(output_filename, 'wb') as file:
    pickle.dump(rf_model, file)
print(f"\nSuccess! Random Forest model saved as '{output_filename}'.")
