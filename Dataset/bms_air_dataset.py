import requests
import zipfile
import io
import os
from pathlib import Path
import pandas as pd
import glob



def download_and_unzip_bms_data(dest_path: str = "./bms_air_quality_data"):
    """
    Downloads and unzips the Beijing Multi-Site Air Quality dataset.
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00501/PRSA2017_Data_20130301-20170228.zip"

    dest_path = Path(dest_path)
    if dest_path.exists():
        print(f"Data directory '{dest_path}' already exists. Skipping download.")
        return

    print(f"Downloading data from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an exception for bad status codes

    # Unzip in memory
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        print(f"Extracting files to '{dest_path}'...")
        z.extractall(dest_path)

    print("Download and extraction complete.")


# Run the download function
download_and_unzip_bms_data()

def load_all_stations(data_dir: str = "./bms_air_quality_data/PRSA_Data_20130301-20170228"):
    """
    Loads all station CSVs into a single pandas DataFrame.
    """
    csv_files = glob.glob(f"{data_dir}/*.csv")
    if not csv_files:
        raise FileNotFoundError("No CSV files found. Make sure you've downloaded the data.")

    df_list = []
    for f in csv_files:
        # Load the CSV
        df_station = pd.read_csv(f)
        # The station name is in the 'station' column, which is constant per file
        station_name = df_station['station'].iloc[0]
        print(f"Loaded data for station: {station_name}")
        df_list.append(df_station)

    # Combine all dataframes into one
    full_df = pd.concat(df_list, ignore_index=True)
    return full_df


# Load the data
df = load_all_stations()
print("\nCombined DataFrame shape:", df.shape)
print("Combined DataFrame head:")
print(df.head())


def clean_bms_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the combined BMS Air Quality DataFrame.
    """
    # 1. Create a proper datetime column
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])

    # 2. Drop redundant columns
    df = df.drop(['No', 'year', 'month', 'day', 'hour'], axis=1)

    # 3. Handle numeric conversions and missing values
    # We include 'wd' (wind direction) here as it's a feature to be filled.
    cols_to_process = [
        'PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP',
        'PRES', 'DEWP', 'RAIN', 'WSPM', 'wd'
    ]

    # We separate numeric from object columns for safe processing
    numeric_cols = [col for col in cols_to_process if col != 'wd']

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 4. Handle remaining missing values with the robust method.
    # We explicitly select the columns to fill, leaving 'station' untouched.
    cols_to_fill = [col for col in df.columns if col != 'station']
    df[cols_to_fill] = df.groupby('station')[cols_to_fill].ffill()

    return df


# Clean the dataframe
df_clean = clean_bms_data(df.copy())

print("\nCleaned DataFrame head:")
print(df_clean.head())


# Pivot the table, explicitly defining the index and columns
df_panel = df_clean.pivot(index='datetime', columns='station')

# Let's reorder the columns to be (station, feature) for better readability
df_panel = df_panel.swaplevel(0, 1, axis=1).sort_index(axis=1)

print("\n✅ Final Panel DataFrame shape:", df_panel.shape)
print("Final Panel DataFrame head:")
print(df_panel.head())
print("\nExample of irregularity (NaN values where stations were offline):")
# Display a slice where data is likely to be missing for some stations
print(df_panel.loc['2017-02-25':'2017-02-28', pd.IndexSlice[:, 'PM2.5']])