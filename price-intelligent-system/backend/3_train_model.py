import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error

print("Loading Master Dataset...")
df = pd.read_csv('data/master_dataset.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# --- 1. Feature Engineering (Making the AI Smart) ---
print("Generating economic indicators (Lags and Rolling Averages)...")
# How much was it yesterday? 3 days ago? A week ago?
for lag in [1, 3, 7]:
    df[f'Price_Lag_{lag}'] = df['Modal_Price'].shift(lag)

# What is the momentum? (7-day and 30-day moving averages)
df['Rolling_Mean_7d'] = df['Modal_Price'].shift(1).rolling(window=7).mean()
df['Rolling_Mean_30d'] = df['Modal_Price'].shift(1).rolling(window=30).mean()

# Is it harvest season or Diwali? (Extracting the month)
df['Month'] = df['Date'].dt.month

# --- 2. Define the Target (What we want to predict) ---
# We want to predict the price exactly 7 days into the future
df['Target_Price_7d'] = df['Modal_Price'].shift(-7)

# Drop rows that have NaNs because of our shifting/lagging
df.dropna(inplace=True)

# --- 3. The Time-Series Split ---
# We train the AI on 2021, 2022, and 2023. 
# We test it on 2024 to see if it can predict an unseen year.
print("Splitting Data: Train (2021-2023) | Test (2024)...")
train_df = df[df['Date'].dt.year <= 2023]
test_df = df[df['Date'].dt.year == 2024]

# These are the columns the AI is allowed to look at
features = [
    'Temperature_C', 'Rainfall_mm', 'Arrivals_Tonnes', 'Diesel_Price_Rs', 
    'Modal_Price', 'Price_Lag_1', 'Price_Lag_3', 'Price_Lag_7', 
    'Rolling_Mean_7d', 'Rolling_Mean_30d', 'Month'
]

X_train, y_train = train_df[features], train_df['Target_Price_7d']
X_test, y_test = test_df[features], test_df['Target_Price_7d']

# --- 4. Train the XGBoost Engine ---
print("Training XGBoost Regressor...")
model = xgb.XGBRegressor(
    n_estimators=300,        # Number of decision trees
    learning_rate=0.05,      # How fast it learns
    max_depth=5,             # Complexity of the trees
    subsample=0.8,           # Prevents overfitting
    random_state=42
)
model.fit(X_train, y_train)

# --- 5. Evaluate the AI ---
predictions = model.predict(X_test)
mape = mean_absolute_percentage_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

print("\n=== MODEL PERFORMANCE (2024 Unseen Data) ===")
print(f"Mean Absolute Error (MAE): Rs. {mae:.2f} per quintal")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2%}")
print("============================================")

# --- 6. Save Model ---
with open('models/xgboost_model.pkl', 'wb') as file:
    pickle.dump(model, file)
print("\nSuccess! Model saved as 'xgboost_model.pkl'. Ready for the Streamlit Dashboard.")
