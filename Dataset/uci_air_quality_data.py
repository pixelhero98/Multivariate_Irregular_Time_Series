import pandas as pd
import numpy as np
import requests
import zipfile
import io

# (The two functions download_and_load_uci_air_quality and clean_uci_air_quality are the same)
def download_and_load_uci_air_quality():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip"
    print(f"Downloading data from {url}...")
    response = requests.get(url)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        excel_filename = next(f for f in z.namelist() if f.endswith('.xlsx'))
        print(f"Reading '{excel_filename}' from the archive...")
        with z.open(excel_filename) as f:
            df = pd.read_excel(f, na_values=[-200])
    print("Load complete.")
    return df

def clean_uci_air_quality(df: pd.DataFrame) -> pd.DataFrame:
    df['datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
    df = df.set_index('datetime')
    df = df.drop(['Date', 'Time'], axis=1)
    df = df.dropna(axis=1, how='all')
    df = df.interpolate(method='time')
    return df

# --- Run the full process ---
df_uci_raw = download_and_load_uci_air_quality()
df_uci = clean_uci_air_quality(df_uci_raw.copy())

print("\n✅ UCI Air Quality processing complete. Final DataFrame info:")
df_uci.info()
print("\nFinal DataFrame head:")
print(df_uci.head())

# --- NEW: Save the cleaned DataFrame to a CSV file ---
output_filename = 'uci_air_quality_cleaned.csv'
df_uci.to_csv(output_filename)
print(f"\n✅ Final cleaned data saved to '{output_filename}' in your project directory.")