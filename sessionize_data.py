
import pandas as pd
from tqdm import tqdm

def create_sessions_by_timeout(input_csv_path, output_csv_path, timeout_seconds=1800):
    """
    Reads the processed log data and segments it into distinct user sessions based on an inactivity timeout.

    - Loads the `bu_events_processed.csv` file.
    - Groups events by the original `visitorid` (client machine).
    - For each client's event stream, if the time between two consecutive events exceeds
      `timeout_seconds`, it assigns a new, unique session ID.
    - Saves the result to a new CSV with a `session_id` column.
    """
    print(f"--- Creating sessions from {input_csv_path} with a {timeout_seconds/60}-minute timeout ---")
    try:
        df = pd.read_csv(input_csv_path, low_memory=False)
    except FileNotFoundError:
        print(f"ERROR: Input file not found at '{input_csv_path}'. Please run previous scripts first.")
        return

    # Ensure data is sorted correctly
    df.sort_values(by=['visitorid', 'timestamp'], inplace=True)

    # Calculate the time difference between consecutive events for each visitor
    df['time_diff'] = df.groupby('visitorid')['timestamp'].diff()

    # A new session starts if the time difference is greater than the timeout
    # The first event for any visitor also starts a new session
    df['new_session_marker'] = (df['time_diff'] > timeout_seconds) | (df['time_diff'].isna())

    # Use cumsum() to create a unique session ID for each identified session
    df['session_id'] = df['new_session_marker'].cumsum()

    # --- Finalizing the DataFrame ---
    # Select and rename columns for clarity
    output_df = df[['session_id', 'timestamp', 'itemid']].copy()
    output_df.rename(columns={'itemid': 'url'}, inplace=True)

    # --- Reporting Results ---
    num_original_visitors = df['visitorid'].nunique()
    num_created_sessions = output_df['session_id'].nunique()

    print(f"\nOriginal number of unique clients: {num_original_visitors:,}")
    print(f"New number of distinct sessions created: {num_created_sessions:,}")

    print(f"\nSaving sessionized data to '{output_csv_path}'...")
    output_df.to_csv(output_csv_path, index=False)

    print("\nSessionization complete.")

if __name__ == '__main__':
    INPUT_CSV = 'bu_events_processed.csv'
    OUTPUT_CSV = 'bu_sessions.csv'
    # Using a 30-minute (1800 seconds) timeout as a standard for web sessions
    create_sessions_by_timeout(INPUT_CSV, OUTPUT_CSV, timeout_seconds=1800)

    # --- Quick analysis of the new session lengths ---
    print("\n--- Analyzing new session lengths ---")
    sessions_df = pd.read_csv(OUTPUT_CSV)
    session_lengths = sessions_df.groupby('session_id').size()
    print(session_lengths.describe())
