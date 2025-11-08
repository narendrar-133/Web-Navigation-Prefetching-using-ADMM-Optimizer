
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
    - Extracts bytes for bandwidth analysis.
    - Filters out common non-content file types (images, css, etc.).
    - Converts the data into a structured DataFrame.
    - Saves the result to a CSV file.
    """
    print(f"Starting to process logs in '{root_directory}'...")
    all_events = []
    
    ignored_extensions = ('.gif', '.jpg', '.jpeg', '.png', '.css', '.ico', '.js', '.xbm')

    file_paths = []
    for subdir, _, files in os.walk(root_directory):
        for file in files:
            file_paths.append(os.path.join(subdir, file))

    for file_path in tqdm(file_paths, desc="Processing log files"):
        try:
            with open(file_path, 'r', errors='ignore') as f:
                for line in f:
                    try:
                        parts = line.strip().split()
                        if len(parts) == 6:
                            client_id = parts[0]
                            timestamp = parts[1]
                            url = parts[3]
                            
                            # Attempt to parse bytes, default to 0 on failure
                            try:
                                bytes_sent = int(parts[4])
                            except ValueError:
                                bytes_sent = 0

                            if not url.lower().endswith(ignored_extensions):
                                all_events.append({
                                    'visitorid': client_id,
                                    'timestamp': timestamp,
                                    'itemid': url,
                                    'bytes': bytes_sent  # Include bytes in the record
                                })
                    except IndexError:
                        continue
        except Exception as e:
            print(f"Could not read file {file_path}: {e}")

    if not all_events:
        print("CRITICAL WARNING: No events were processed.")
        return

    print(f"\nSuccessfully parsed {len(all_events):,} events.")
    
    df = pd.DataFrame(all_events)
    
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df.dropna(subset=['timestamp'], inplace=True)
    df['timestamp'] = df['timestamp'].astype(int)
    
    print("Sorting data by visitor and timestamp...")
    df.sort_values(by=['visitorid', 'timestamp'], inplace=True)

    print(f"Saving cleaned data (with bytes) to '{output_csv_path}'...")
    df.to_csv(output_csv_path, index=False)
    
    print("\nData preparation complete.")

if __name__ == '__main__':
    DATA_ROOT = 'BU-www-client-traces/condensed/'
    OUTPUT_CSV = 'bu_events_processed.csv'
    parse_and_clean_bu_logs(DATA_ROOT, OUTPUT_CSV)
