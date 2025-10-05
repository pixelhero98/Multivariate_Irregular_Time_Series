import isd
import pandas as pd
from tqdm.auto import tqdm # For a nice progress bar


# Get a list of all stations in the ISD inventory
stations = isd.get_stations()

# Filter for stations in Great Britain (country code 'UK') that were active in 2022
uk_stations_2022 = stations[
    (stations['country'] == 'UK') &
    (stations['begin'] <= '2022-01-01') &
    (stations['end'] >= '2022-12-31')
]

# Get the list of station IDs to download
station_ids = uk_stations_2022['usaf'] + uk_stations_2022['wban']
print(f"Found {len(station_ids)} active stations in Great Britain for 2022.")
print("Example station IDs:", station_ids.head().tolist())


def download_isd_subset(station_ids, start_year, end_year):
    """
    Downloads data for a list of station IDs over a year range.
    """
    df_list = []

    # Use tqdm to show progress as this can be slow
    for station_id in tqdm(station_ids, desc="Downloading station data"):
        try:
            # The library fetches data year by year
            for year in range(start_year, end_year + 1):
                df_station, _ = isd.get_data(station_id, year)
                df_list.append(df_station)
        except Exception as e:
            print(f"Could not fetch data for station {station_id}: {e}")

    if not df_list:
        raise ValueError("No data could be downloaded for the selected stations.")

    return pd.concat(df_list, ignore_index=True)


# Download the data for our selected stations for the year 2022
df_raw = download_isd_subset(station_ids.tolist(), 2022, 2022)

print("\nCombined DataFrame shape:", df_raw.shape)
print("Combined DataFrame head:")
print(df_raw.head())


def clean_and_panel_isd_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and pivots the raw ISD data into a multivariate panel.
    """
    # 1. Select a subset of useful feature channels
    feature_cols = [
        'station', 'date', 'temperature', 'dew_point',
        'sea_level_pressure', 'wind_speed', 'precipitation'
    ]
    df = df[feature_cols]

    # 2. Create a proper datetime index
    df['datetime'] = pd.to_datetime(df['date'])
    df = df.set_index('datetime').drop('date', axis=1)

    # 3. Handle missing values using forward-fill within each station group
    # The library already uses NaNs for missing data
    df = df.groupby('station').ffill()

    # 4. Pivot to create the final panel format
    df_panel = df.pivot(columns='station')

    # Reorder columns for better readability
    df_panel = df_panel.swaplevel(0, 1, axis=1).sort_index(axis=1)

    return df_panel


# Process the raw data
df_panel = clean_and_panel_isd_data(df_raw.copy())

print("\n✅ Processing Complete! Final Panel DataFrame head:")
print(df_panel.head())
print("\nExample of irregularity (NaN values where stations were offline):")
print(df_panel.loc['2022-07-01 00:00:00':'2022-07-01 05:00:00', pd.IndexSlice[:, 'temperature']])
