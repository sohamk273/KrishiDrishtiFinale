import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

print("=== Training Random Forest Regressor for Onion 7-Day Forecast ===")
print("Loading Master Dataset...")
df = pd.read_csv('master_dataset.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# --- 1. Feature Engineering (Identical to 3_train_model.py) ---
print("Generating economic indicators (Lags and Rolling Averages)...")
for lag in [1, 3, 7]:
    df[f'Price_Lag_{lag}'] = df['Modal_Price'].shift(lag)

df['Rolling_Mean_7d'] = df['Modal_Price'].shift(1).rolling(window=7).mean()
df['Rolling_Mean_30d'] = df['Modal_Price'].shift(1).rolling(window=30).mean()
df['Month'] = df['Date'].dt.month

# --- 2. Define the Target (7-day forecast horizon) ---
df['Target_Price_7d'] = df['Modal_Price'].shift(-7)
df.dropna(inplace=True)

# --- 3. Chronological Time-Series Split (Train <= 2023, Test == 2024) ---
print("Splitting Data: Train (2021-2023) | Test (2024)...")
train_df = df[df['Date'].dt.year <= 2023]
test_df = df[df['Date'].dt.year == 2024]

# Exact same 11 features as 3_train_model.py and app.py legacy onion
features = [
    'Temperature_C', 'Rainfall_mm', 'Arrivals_Tonnes', 'Diesel_Price_Rs', 
    'Modal_Price', 'Price_Lag_1', 'Price_Lag_3', 'Price_Lag_7', 
    'Rolling_Mean_7d', 'Rolling_Mean_30d', 'Month'
]

X_train, y_train = train_df[features], train_df['Target_Price_7d']
X_test, y_test = test_df[features], test_df['Target_Price_7d']

# --- 4. Train the Random Forest Regressor ---
print("Training RandomForestRegressor (n_estimators=300, random_state=42, n_jobs=-1)...")
rf_model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# --- 5. Evaluate Random Forest on 2024 Test Data ---
rf_preds = rf_model.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_preds)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
rf_r2 = r2_score(y_test, rf_preds)
rf_mape = mean_absolute_percentage_error(y_test, rf_preds)

print("\n=== RANDOM FOREST MODEL PERFORMANCE (2024 Unseen Data) ===")
print(f"Mean Absolute Error (MAE): Rs. {rf_mae:.2f} per quintal")
print(f"Root Mean Squared Error (RMSE): Rs. {rf_rmse:.2f} per quintal")
print(f"R-squared (R²): {rf_r2:.4f}")
print(f"Mean Absolute Percentage Error (MAPE): {rf_mape:.2%}")
print("==========================================================")

# --- 6. Save Random Forest Model ---
output_filename = 'randomforest_model.pkl'
with open(output_filename, 'wb') as file:
    pickle.dump(rf_model, file)
print(f"\nSuccess! Random Forest model saved as '{output_filename}'. Ready for Streamlit Dashboard.")
