import pandas as pd

def remove_dataset_redundancy(input_filename, output_filename='cleaned_dataset.csv'):
    print(f"Loading {input_filename}...")
    df = pd.read_csv(input_filename)
    
    # 1. Convert Date to standard datetime and sort chronologically
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.sort_values('Date')
    
    initial_row_count = len(df)
    print(f"Initial row count: {initial_row_count}")
    
    # 2. Group by Date to resolve duplicate entries/varieties on the same day
    # We aggregate sensibly: sum arrivals, average modal prices, take first for weather & diesel
    agg_rules = {
        'Temperature_C': 'first',
        'Rainfall_mm': 'first',
        'Arrivals_Tonnes': 'sum',
        'Min_Price': 'min',
        'Max_Price': 'max',
        'Modal_Price': 'mean',
        'Diesel_Price_Rs': 'first'
    }
    
    # Keep only columns that actually exist in the CSV to avoid key errors
    active_agg_rules = {k: v for k, v in agg_rules.items() if k in df.columns}
    
    df_clean = df.groupby('Date').agg(active_agg_rules).reset_index()
    
    # 3. Forward-fill any minor missing gaps or NaNs
    df_clean = df_clean.ffill().bfill()
    
    # 4. Save clean non-redundant dataset
    df_clean.to_csv(output_filename, index=False)
    
    final_row_count = len(df_clean)
    print(f"Successfully cleaned! Rows reduced from {initial_row_count} to {final_row_count} unique daily records.")
    print(f"Saved clean file as: {output_filename}\n")

# Run the function on your Tur Dal dataset (or any other dataset)
if __name__ == "__main__":
    remove_dataset_redundancy('master_dataset_latur.csv', 'master_dataset_latur.csv')
