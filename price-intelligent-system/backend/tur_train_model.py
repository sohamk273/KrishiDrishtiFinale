import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
import pickle

print("Loading Tur Dal Master Dataset...")
df = pd.read_csv('data/master_dataset_latur.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# Clean duplicate dates by daily aggregation
agg_funcs = {
    'Temperature_C': 'first',
    'Rainfall_mm': 'first',
    'Arrivals_Tonnes': 'sum',
    'Min_Price': 'mean',
    'Max_Price': 'mean',
    'Modal_Price': 'mean',
    'Diesel_Price_Rs': 'first'
}
df = df.groupby('Date').agg(agg_funcs).reset_index()

# --- 1. Elite Feature Engineering ---
print("Generating multi-horizon momentum and volatility features...")
for lag in [1, 2, 3, 4, 5, 7, 10, 14, 21]:
    df[f'Price_Lag_{lag}'] = df['Modal_Price'].shift(lag)

df['Rolling_Mean_3d'] = df['Modal_Price'].shift(1).rolling(window=3).mean()
df['Rolling_Mean_7d'] = df['Modal_Price'].shift(1).rolling(window=7).mean()
df['Rolling_Mean_14d'] = df['Modal_Price'].shift(1).rolling(window=14).mean()
df['Rolling_Std_7d'] = df['Modal_Price'].shift(1).rolling(window=7).std()

df['Month'] = df['Date'].dt.month
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['DayOfYear'] = df['Date'].dt.dayofyear

# --- 2. Define Target ---
df['Target_Price_7d'] = df['Modal_Price'].shift(-7)
df.dropna(inplace=True)

# --- 3. Time-Series Split (Train on 2021-2023, Test on 2024) ---
train_df = df[df['Date'].dt.year <= 2023]
test_df = df[df['Date'].dt.year == 2024]

features = [
    'Temperature_C', 'Rainfall_mm', 'Arrivals_Tonnes', 'Diesel_Price_Rs', 
    'Modal_Price', 'Price_Lag_1', 'Price_Lag_2', 'Price_Lag_3', 'Price_Lag_4',
    'Price_Lag_5', 'Price_Lag_7', 'Price_Lag_10', 'Price_Lag_14', 'Price_Lag_21', 
    'Rolling_Mean_3d', 'Rolling_Mean_7d', 'Rolling_Mean_14d', 'Rolling_Std_7d', 
    'Month', 'DayOfWeek', 'DayOfYear'
]

X_train, y_train = train_df[features], train_df['Target_Price_7d']
X_test, y_test = test_df[features], test_df['Target_Price_7d']

# --- 4. High-Performance Training ---
print("Training Tur Dal Elite Gradient Boosting Regressor...")
model = GradientBoostingRegressor(
    n_estimators=400,
    learning_rate=0.01,
    max_depth=6,
    subsample=0.8,
    random_state=42
)
model.fit(X_train, y_train)

# --- 5. Evaluate Accuracy ---
predictions = model.predict(X_test)
mape = mean_absolute_percentage_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
accuracy = (1 - mape) * 100

print("\n=== TUR DAL MODEL PERFORMANCE ===")
print(f"Mean Absolute Error (MAE): Rs. {mae:.2f} per quintal")
print(f"Model Accuracy (1 - MAPE): {accuracy:.2f}%")
print("==================================")

# --- 6. Save Model ---
with open('models/turdal_model.pkl', 'wb') as file:
    pickle.dump(model, file)
print("\nSuccess! Saved 'turdal_model.pkl'.")
