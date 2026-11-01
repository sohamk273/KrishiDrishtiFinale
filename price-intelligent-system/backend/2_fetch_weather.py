import pandas as pd
import openmeteo_requests
import numpy as np
import time

print("Loading cleaned price data...")
prices_df = pd.read_csv('lasalgaon_onions.csv')
prices_df['Date'] = pd.to_datetime(prices_df['Date'])

# Automatically detect the date range from your CSV
start_date = prices_df['Date'].min().strftime('%Y-%m-%d')
end_date = prices_df['Date'].max().strftime('%Y-%m-%d')

print(f"Fetching weather for Nashik from {start_date} to {end_date}...")

# 1. Setup Open-Meteo API
openmeteo = openmeteo_requests.Client()
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 20.00,
    "longitude": 73.78,
    "start_date": start_date,
    "end_date": end_date,
    "daily": ["temperature_2m_mean", "precipitation_sum"],
    "timezone": "Asia/Kolkata"
}

# 2. THE FIX: Robust Retry Loop for API Timeouts
max_retries = 3
daily = None

for attempt in range(max_retries):
    try:
        print(f"API Request Attempt {attempt + 1}/{max_retries}...")
        responses = openmeteo.weather_api(url, params=params)
        daily = responses[0].Daily()
        print("Weather data successfully downloaded!")
        break # Exit the loop if successful
    except Exception as e:
        if attempt < max_retries - 1:
            print("Server timed out. Retrying in 5 seconds...")
            time.sleep(5)
        else:
            print(f"Error: Open-Meteo API is unresponsive. ({e})")
            exit()

# 3. Build the Weather Dataset
weather_df = pd.DataFrame({
    "Date": pd.date_range(
        start=pd.to_datetime(daily.Time(), unit="s", utc=True),
        end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=daily.Interval()),
        inclusive="left"
    ).normalize(),
    "Temperature_C": daily.Variables(0).ValuesAsNumpy(),
    "Rainfall_mm": daily.Variables(1).ValuesAsNumpy()
})

# Make Date column timezone-naive to match our CSV dates
weather_df['Date'] = weather_df['Date'].dt.tz_localize(None)

# 4. Merge Weather with Prices
print("Merging weather and handling non-trading days...")
master_df = pd.merge(weather_df, prices_df, on='Date', how='outer')
master_df = master_df.sort_values('Date')

# Forward-fill: If Sunday is missing, use Saturday's closing price
cols_to_fill = ['Arrivals_Tonnes', 'Min_Price', 'Max_Price', 'Modal_Price']
master_df[cols_to_fill] = master_df[cols_to_fill].ffill()
master_df.dropna(subset=cols_to_fill, inplace=True)

# 5. Inject Diesel Prices
print("Injecting Transportation Costs (Diesel)...")
diesel_df = pd.read_csv('fuel/maharashtra_diesel_prices.csv')
diesel_df['Date'] = pd.to_datetime(diesel_df['Date'])

master_df = pd.merge(master_df, diesel_df, on='Date', how='left')

# 6. Save the final ML-ready dataset
master_df.to_csv('data/master_dataset.csv', index=False)
print(f"Success! Saved master_dataset.csv with {len(master_df)} continuous rows.")
