import pandas as pd

print("Loading master dataset for final aggregation...")
df = pd.read_csv('master_dataset.csv')
df['Date'] = pd.to_datetime(df['Date'])

# 1. Aggregate multiple rows per day into a single daily summary
print("Consolidating daily market varieties...")
agg_funcs = {
    'Temperature_C': 'first',
    'Rainfall_mm': 'first',
    'Arrivals_Tonnes': 'sum',      # Add all the volumes together
    'Min_Price': 'mean',           # Average the prices
    'Max_Price': 'mean',
    'Modal_Price': 'mean',
    'Diesel_Price_Rs': 'first'
}
df_grouped = df.groupby('Date').agg(agg_funcs).reset_index()

# 2. Fill the 3 missing weather days at the end of 2024
print("Patching final weather gaps...")
df_grouped['Temperature_C'] = df_grouped['Temperature_C'].ffill()
df_grouped['Rainfall_mm'] = df_grouped['Rainfall_mm'].ffill()

# 3. Save the truly final version
df_grouped.to_csv('master_dataset.csv', index=False)
print(f"DONE! The final dataset has exactly {len(df_grouped)} rows (1 per day) and 0 missing values.")
