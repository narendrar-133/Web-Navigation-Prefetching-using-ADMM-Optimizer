
import pandas as pd
from tqdm import tqdm

def create_sessions_by_timeout(input_csv_path, output_csv_path, timeout_seconds=1800):
    """
    Reads the processed log data and segments it into distinct user sessions based on an inactivity timeout.

    - Loads the `bu_events_processed.csv` file (which must include a 'bytes' column).
    - Groups events by the original `visitorid`.
    - Segments sessions based on inactivity timeout.
    - Carries the 'bytes' column through to the output.
    - Saves the result to a new CSV with session_id, timestamp, url, and bytes.
    """
    print(f"--- Creating sessions from {input_csv_path} with a {timeout_seconds/60}-minute timeout ---")
    try:
        # Make sure to read the bytes column
        df = pd.read_csv(input_csv_path, low_memory=False)
        if 'bytes' not in df.columns:
            print("CRITICAL ERROR: The input CSV does not contain the required 'bytes' column.")
            print("Please run the updated 'prepare_bu_data.py' script first.")
            return
    except FileNotFoundError:
        print(f"ERROR: Input file not found at '{input_csv_path}'. Please run previous scripts first.")
        return

    df.sort_values(by=['visitorid', 'timestamp'], inplace=True)

    df['time_diff'] = df.groupby('visitorid')['timestamp'].diff()

    df['new_session_marker'] = (df['time_diff'] > timeout_seconds) | (df['time_diff'].isna())

    df['session_id'] = df['new_session_marker'].cumsum()

    # --- Finalizing the DataFrame ---
    # Include the 'bytes' column in the output
    output_df = df[['session_id', 'timestamp', 'itemid', 'bytes']].copy()
    output_df.rename(columns={'itemid': 'url'}, inplace=True)

    # --- Reporting Results ---
    num_original_visitors = df['visitorid'].nunique()
    num_created_sessions = output_df['session_id'].nunique()

    print(f"\nOriginal number of unique clients: {num_original_visitors:,}")
    print(f"New number of distinct sessions created: {num_created_sessions:,}")

    print(f"\nSaving sessionized data (with bytes) to '{output_csv_path}'...")
    output_df.to_csv(output_csv_path, index=False)

    print("\nSessionization complete.")

if __name__ == '__main__':
    INPUT_CSV = 'bu_events_processed.csv'
    OUTPUT_CSV = 'bu_sessions_markov.csv'
    create_sessions_by_timeout(INPUT_CSV, OUTPUT_CSV, timeout_seconds=1800)

    # --- Quick analysis of the new session lengths ---
    print("\n--- Analyzing new session lengths ---")
    sessions_df = pd.read_csv(OUTPUT_CSV)
    session_lengths = sessions_df.groupby('session_id').size()
    print(session_lengths.describe())
