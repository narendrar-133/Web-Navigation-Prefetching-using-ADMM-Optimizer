
import os
import pandas as pd
from tqdm import tqdm

def parse_and_clean_bu_logs(root_directory, output_csv_path):
    """
    Parses the Boston University server log dataset, cleans it, and saves it
    to a single CSV file.

    - Traverses all subdirectories of the given root directory.
    - Reads each log file line by line.
    - Parses the 6-column format: [client_id] [timestamp] [request_time] [URL] [bytes] [response_time]
    - Filters out common non-content file types (images, css, etc.).
    - Converts the data into a structured DataFrame with 'visitorid', 'timestamp', and 'itemid'.
    - Saves the result to a CSV file.
    """
    print(f"Starting to process logs in '{root_directory}'...")
    all_events = []
    
    # Define file extensions to ignore, as they don't represent user intent
    ignored_extensions = ('.gif', '.jpg', '.jpeg', '.png', '.css', '.ico', '.js', '.xbm')

    # Use os.walk to go through all nested directories
    file_paths = []
    for subdir, _, files in os.walk(root_directory):
        for file in files:
            file_paths.append(os.path.join(subdir, file))

    # Using tqdm for a progress bar
    for file_path in tqdm(file_paths, desc="Processing log files"):
        try:
            with open(file_path, 'r', errors='ignore') as f:
                for line in f:
                    try:
                        parts = line.strip().split()
                        # Ensure the line has the expected structure
                        if len(parts) == 6:
                            client_id = parts[0]
                            timestamp = parts[1]
                            url = parts[3]

                            # Filter out requests to ignored file types
                            if not url.lower().endswith(ignored_extensions):
                                all_events.append({
                                    'visitorid': client_id,
                                    'timestamp': timestamp,
                                    'itemid': url
                                })
                    except IndexError:
                        # This will skip any malformed lines within a file
                        continue
        except Exception as e:
            print(f"Could not read file {file_path}: {e}")

    if not all_events:
        print("CRITICAL WARNING: No events were processed. Check the directory and file format.")
        return

    print(f"\nSuccessfully parsed {len(all_events):,} events.")
    
    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(all_events)
    
    # Convert timestamp to a numeric type for sorting, handling non-numeric values
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df.dropna(subset=['timestamp'], inplace=True)
    df['timestamp'] = df['timestamp'].astype(int)
    
    # Sort the data by user and time
    print("Sorting data by visitor and timestamp...")
    df.sort_values(by=['visitorid', 'timestamp'], inplace=True)

    print(f"Saving cleaned data to '{output_csv_path}'...")
    df.to_csv(output_csv_path, index=False)
    
    print("\nData preparation complete.")
    print(f"Total Unique Visitors: {df['visitorid'].nunique():,}")
    print(f"Total Unique Items (URLs): {df['itemid'].nunique():,}")


if __name__ == '__main__':
    # Define the root directory of the dataset and the desired output file
    DATA_ROOT = 'BU-www-client-traces/condensed/'
    OUTPUT_CSV = 'bu_events_processed.csv'
    parse_and_clean_bu_logs(DATA_ROOT, OUTPUT_CSV)
