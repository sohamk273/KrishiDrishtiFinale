import pandas as pd

print("Loading Kaggle Fuel Dataset...")
# 1. Read the raw fuel dataset
df = pd.read_csv('india_fuel_prices_daily.csv')

# 2. Filter for Mumbai (our transport proxy for Lasalgaon/Nashik)
mumbai_df = df[df['city'].str.lower() == 'mumbai'].copy()

# 3. Format Date and rename columns
mumbai_df['Date'] = pd.to_datetime(mumbai_df['date'])
mumbai_df = mumbai_df.rename(columns={'diesel_price_inr_per_litre': 'Diesel_Price_Rs'})
mumbai_df = mumbai_df[['Date', 'Diesel_Price_Rs']]

# 4. Generate a perfect calendar covering Jan 1, 2021 to Dec 31, 2024
print("Generating complete timeline (2021-01-01 to 2024-12-31)...")
full_dates = pd.DataFrame({
    'Date': pd.date_range(start='2021-01-01', end='2024-12-31', freq='D')
})

# 5. Merge the calendar with the Kaggle data
merged_df = pd.merge(full_dates, mumbai_df, on='Date', how='left')

# 6. The "Pro" Fix: Extrapolate Jan-Apr 2021
print("Interpolating missing historical data for Jan - Apr 2021...")
# Anchor the starting price for Jan 1, 2021
merged_df.loc[merged_df['Date'] == '2021-01-01', 'Diesel_Price_Rs'] = 80.20

# Draw a smooth line between our 80.20 anchor and the May 1st starting price
merged_df['Diesel_Price_Rs'] = merged_df['Diesel_Price_Rs'].interpolate(method='linear')

# Safety net: Forward-fill and backward-fill any remaining random missing days
merged_df['Diesel_Price_Rs'] = merged_df['Diesel_Price_Rs'].ffill().bfill()

# Round to 2 decimal places to keep it clean
merged_df['Diesel_Price_Rs'] = merged_df['Diesel_Price_Rs'].round(2)

# 7. Save the output
merged_df.to_csv('maharashtra_diesel_prices.csv', index=False)
print(f"Done! Saved 'maharashtra_diesel_prices.csv' with exactly {len(merged_df)} continuous rows.")
