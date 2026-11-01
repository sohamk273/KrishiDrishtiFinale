import pandas as pd
import glob
import numpy as np

print("Locating downloaded CSV files...")
# FIX 1: Look for .csv files instead of .xls
file_list = glob.glob("*.csv") 

if not file_list:
    print("Error: No CSV files found in the folder. Please ensure you exported them.")
    exit()

all_cleaned_data = []

for file in file_list:
    print(f"Processing {file}...")
    try:
        # FIX 2: Use read_csv, and set header=None so the messy top rows don't break the columns
        raw_df = pd.read_csv(file, header=None, skiprows=3, on_bad_lines='skip')
        
        # Create a new column 'Market_Name'
        raw_df['Market_Name'] = np.nan
        
        current_market = "Unknown"
        
        # Iterate through the rows to extract the market names
        for index, row in raw_df.iterrows():
            first_col_val = str(row.iloc[0])
            if 'Market Name' in first_col_val:
                current_market = first_col_val.replace('Market Name :', '').strip()
            
            raw_df.at[index, 'Market_Name'] = current_market

        # Rename columns to match the 6 columns from the CSV + our new Market_Name column
        raw_df.columns = ['Arrival_Date', 'Arrivals_Tonnes', 'Variety', 'Min_Price', 'Max_Price', 'Modal_Price', 'Market_Name']
        
        # Drop rows that are just headers or garbage text
        clean_df = raw_df[~raw_df['Arrival_Date'].astype(str).str.contains('Market|Arrival|nan|Date|Commodity', na=False, case=False)].copy()
        
        # Convert date to proper format
        clean_df['Date'] = pd.to_datetime(clean_df['Arrival_Date'], format='%d/%m/%Y', errors='coerce')
        
        # Drop rows where Date conversion failed
        clean_df = clean_df.dropna(subset=['Date'])
        
        all_cleaned_data.append(clean_df)

    except Exception as e:
        print(f"Error processing {file}: {e}")

# Combine all months
print("\nCombining all months...")
master_df = pd.concat(all_cleaned_data, ignore_index=True)

# Filter ONLY for Lasalgaon
print("Filtering for Lasalgaon market...")
lasalgaon_df = master_df[master_df['Market_Name'].str.contains('LASALGAON', case=False, na=False)].copy()

# Sort chronologically and save
lasalgaon_df = lasalgaon_df.sort_values('Date')
lasalgaon_df = lasalgaon_df[['Date', 'Arrivals_Tonnes', 'Min_Price', 'Max_Price', 'Modal_Price']]

lasalgaon_df.to_csv('lasalgaon_onions.csv', index=False)
print(f"Done! Saved lasalgaon_onions.csv with {len(lasalgaon_df)} records.")
