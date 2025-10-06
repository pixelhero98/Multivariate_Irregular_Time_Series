import os
import glob
import pandas as pd
import numpy as np
import requests
import zipfile
import io
from pathlib import Path  # Import Path for robust path handling


def download_physionet_2012(dl_dir: str = 'physionet_data'):
    """
    Downloads the PhysioNet 2012 Challenge data if it doesn't exist locally.
    """
    url = "https://physionet.org/static/published-projects/challenge-2012/physionet-challenge-2012-set-a-1.0.0.zip"
    outcomes_file = os.path.join(dl_dir, 'set-a', 'Outcomes-a.txt')
    if os.path.exists(outcomes_file):
        print("Raw PhysioNet 2012 dataset already exists locally.")
        return

    print(f"Downloading raw PhysioNet 2012 Challenge data from {url}...")
    response = requests.get(url)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall(dl_dir)
    print("Download and extraction complete.")


def process_physionet_features(data_dir: str = 'physionet_data/set-a') -> pd.DataFrame:
    """
    Processes the raw PhysioNet 2012 text files into a single feature DataFrame.
    """
    patient_files = glob.glob(os.path.join(data_dir, "*.txt"))
    patient_files = [f for f in patient_files if 'Outcomes' not in f]
    all_patient_dfs = []

    print(f"Processing {len(patient_files)} raw patient files...")
    for file_path in patient_files:
        df_raw = pd.read_csv(file_path)
        time_parts = df_raw['Time'].str.split(':', expand=True)
        minutes = time_parts[0].astype(int) * 60 + time_parts[1].astype(int)
        df_raw['Minutes'] = minutes
        df_pivot = df_raw.pivot_table(index='Minutes', columns='Parameter', values='Value')
        df_pivot['RecordID'] = df_raw['RecordID'].iloc[0]
        all_patient_dfs.append(df_pivot)

    full_df = pd.concat(all_patient_dfs)
    full_df = full_df.set_index('RecordID', append=True).swaplevel(0, 1).sort_index()
    full_df = full_df.groupby('RecordID').ffill()
    return full_df


def load_physionet_outcomes(outcomes_path: str = 'physionet_data/set-a/Outcomes-a.txt') -> pd.DataFrame:
    """
    Loads the outcomes data (the target variables).
    """
    df_outcomes = pd.read_csv(outcomes_path)
    df_outcomes = df_outcomes.set_index('RecordID')
    return df_outcomes


if __name__ == "__main__":
    # --- Caching Logic ---
    processed_data_path = Path('physionet_2012_processed.parquet')

    if processed_data_path.exists():
        print(f"Loading processed data from cache: '{processed_data_path}'")
        final_df = pd.read_parquet(processed_data_path)
    else:
        print("Processed data cache not found. Running full processing pipeline...")
        # 1. Download the raw data (if needed)
        download_physionet_2012()

        # 2. Process the features from the raw text files
        features_df = process_physionet_features()

        # 3. Load the outcomes (labels)
        outcomes_df = load_physionet_outcomes()

        # 4. Merge features and labels into a final dataset
        final_df = features_df.join(outcomes_df, on='RecordID')

        # 5. Save the processed data to cache for future runs
        print(f"Saving processed data to '{processed_data_path}'...")
        final_df.to_parquet(processed_data_path)

    # --- Display Info (runs every time) ---
    print("\n✅ Processing complete. Final merged DataFrame info:")
    final_df.info()

    print("\nFinal DataFrame head:")
    print(final_df.head())

    print("\nExample of processed data for a single patient (RecordID 140501):")
    # Using .loc can be slow on large multi-index, xs is often faster
    if 140501 in final_df.index.get_level_values('RecordID'):
        print(final_df.xs(140501, level='RecordID').head(10))