import logging

import pandas as pd
import os

# Create a custom formatter with your desired time format
time_format = "%Y-%m-%d %H:%M:%S"
formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s', datefmt=time_format)

# Create a logger and set the custom formatter
logger = logging.getLogger('custom_logger')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

# Set the log level (optional, can be DEBUG, INFO, WARNING, ERROR, CRITICAL)
logger.setLevel(logging.DEBUG)

if __name__ == '__main__':
    time_start = pd.Timestamp.now()
    # Directory where the CSV files are stored
    input_dir = 'data_dyad_monthly'

    # List all CSV files in the directory
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    csv_files = list(sorted(csv_files))

    # Initialize an empty list to store DataFrames
    dfs = []

    # Loop over the list of CSV files
    for file in sorted(csv_files):  # Sorting ensures the files are processed in order
        logger.info(f'Reading {file}')
        file_path = os.path.join(input_dir, file)
        # Read the current CSV into a DataFrame
        df = pd.read_csv(file_path)
        # Append the DataFrame to the list
        dfs.append(df)
    logger.info("All files read.")
    # Concatenate all DataFrames in the list into one
    dyad_df = pd.concat(dfs, ignore_index=True)

    save_to_parquet = True
    if save_to_parquet:
        dyad_df.to_parquet('data_dyad_monthly/dyad_df.parquet')
    logger.info('Data saved to Parquet format.')
    time_end = pd.Timestamp.now()
    time_elapsed = time_end - time_start
    logger.info(f"Time elapsed in minutes: {time_elapsed.total_seconds() / 60:.2f}")
