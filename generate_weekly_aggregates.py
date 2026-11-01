"""
generate_weekly_aggregates.py
=============================
KrishiDrishti – Weekly Modal Price Aggregation Script (2021–2025)

Aggregates historical daily Modal Price observations into ISO calendar weeks:
  1. Onion (Lasalgaon)  : master_dataset.csv
  2. Potato (Achalda)   : master_dataset_achalda.csv
  3. Tur Dal (Latur)    : master_dataset_latur.csv

Saves output CSVs in data/weekly_aggregated/:
  - Year
  - ISO_Week
  - Week_Start_Date
  - Week_End_Date
  - Observation_Count
  - Weekly_Mean_Modal_Price
"""

import os
import pandas as pd

OUTDIR = os.path.join("data", "weekly_aggregated")
os.makedirs(OUTDIR, exist_ok=True)

SOURCES = {
    "onion_lasalgaon":  "master_dataset.csv",
    "potato_achalda":   "master_dataset_achalda.csv",
    "turdal_latur":     "master_dataset_latur.csv",
}

def generate_weekly_data():
    for slug, src in SOURCES.items():
        if not os.path.exists(src):
            print(f"Skipping {src} (not found)")
            continue
        
        df = pd.read_csv(src)
        df["Date"] = pd.to_datetime(df["Date"])

        # Filter strictly to 2021-01-01 to 2025-12-31
        mask = (df["Date"] >= "2021-01-01") & (df["Date"] <= "2025-12-31")
        sub  = df[mask].copy()

        # ISO calendar fields
        iso = sub["Date"].dt.isocalendar()
        sub["ISO_Year"] = iso.year.astype(int)
        sub["ISO_Week"] = iso.week.astype(int)

        # Group by ISO year + week; compute arithmetic mean and observation count
        weekly = (
            sub.groupby(["ISO_Year", "ISO_Week"])
            .agg(
                Observation_Count       = ("Modal_Price", "count"),
                Weekly_Mean_Modal_Price = ("Modal_Price", "mean"),
                _first_date             = ("Date", "min"),
                _last_date              = ("Date", "max"),
            )
            .reset_index()
        )
        weekly.rename(columns={
            "ISO_Year":   "Year",
            "_first_date": "Week_Start_Date",
            "_last_date":  "Week_End_Date",
        }, inplace=True)

        weekly["Weekly_Mean_Modal_Price"] = weekly["Weekly_Mean_Modal_Price"].round(4)
        weekly = weekly.sort_values(["Year", "ISO_Week"]).reset_index(drop=True)

        cols = ["Year", "ISO_Week", "Week_Start_Date", "Week_End_Date",
                "Observation_Count", "Weekly_Mean_Modal_Price"]
        weekly = weekly[cols]

        out_path = os.path.join(OUTDIR, f"weekly_{slug}.csv")
        weekly.to_csv(out_path, index=False)
        print(f"✅ Generated {out_path} ({len(weekly)} ISO weeks from {len(sub)} daily rows)")

if __name__ == "__main__":
    generate_weekly_data()
