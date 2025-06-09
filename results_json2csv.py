import json
import os
import csv
from pathlib import Path
import argparse # Added for command-line arguments

def format_time_from_seconds(seconds_str):
    """Converts seconds string (possibly float) to MM:SS format."""
    seconds = float(seconds_str)
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    return f"{minutes:02d}:{remaining_seconds:02d}"

def process_json_to_csv(json_file_path, output_csv_path, threshold=0.6):
    """
    Parses a JSON file (expected to be results_spotting.json), filters actions, 
    extracts half, time, and action, and saves to CSV.
    Raises errors if issues occur during file operations or data processing.
    """
    print(f"--- Processing JSON file: {json_file_path} ---")
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    filtered_data_for_csv = []
    actions_list = []

    # Attempt to find the list of actions/predictions
    if isinstance(data, list):
        actions_list = data
    elif isinstance(data, dict):
        if "predictions" in data and isinstance(data["predictions"], list):
            actions_list = data["predictions"]
        elif "SoccerNetPredictions" in data and isinstance(data["SoccerNetPredictions"], list):
             actions_list = data["SoccerNetPredictions"]
        elif "annotations" in data and isinstance(data["annotations"], list):
            actions_list = data["annotations"]
        else:
            is_single_action = True
            for key_check in ['label', 'confidence']:
                if key_check not in data:
                    is_single_action = False
                    break
            if is_single_action and ('gameTime' in data or 'position' in data):
                 actions_list = [data]
    
    if not actions_list:
        print(f"No actions list found or identified in {json_file_path}. CSV will not be created for this file.")
        return

    for action in actions_list:
        if not isinstance(action, dict):
            continue

        confidence_val = None
        label_val = None
        half_val = "N/A"
        time_str = "00:00"

        for conf_key in ['confidence', 'score', 'probability']:
            if conf_key in action:
                confidence_val = float(action[conf_key])
                break
        
        for label_key in ['label', 'action', 'class', 'event']:
            if label_key in action:
                label_val = action[label_key]
                break

        if confidence_val is None or label_val is None:
            continue

        if confidence_val >= threshold:
            if 'gameTime' in action and isinstance(action['gameTime'], str) and " - " in action['gameTime']:
                parts = action['gameTime'].split(" - ", 1)
                if len(parts) == 2: # Ensure split produced two parts
                    half_val = parts[0].strip()
                    time_str = parts[1].strip()
                else: # Malformed gameTime, fallback or log
                    print(f"Warning: Malformed gameTime '{action['gameTime']}' in {json_file_path}. Falling back for time.")
                    time_pos_val = None
                    for time_key in ['position', 'time', 'timestamp', 'start_time']:
                        if time_key in action:
                            time_pos_val = action[time_key]
                            break
                    if time_pos_val is not None:
                        time_str = format_time_from_seconds(str(time_pos_val))

            else: # Fallback to 'position' or similar if 'gameTime' is not present/valid
                time_pos_val = None
                for time_key in ['position', 'time', 'timestamp', 'start_time']:
                    if time_key in action:
                        time_pos_val = action[time_key]
                        break
                if time_pos_val is not None:
                    time_str = format_time_from_seconds(str(time_pos_val))

            # Ensure confidence_val is included in the data written to CSV
            filtered_data_for_csv.append({'Half': half_val, 'Time': time_str, 'Action': label_val, 'Confidence': confidence_val})

    if filtered_data_for_csv:
        # Sort the data before writing
        def sort_key(item):
            half = item['Half']
            time_str = item['Time']
            
            # Half sorting: try to convert to int, N/A or others go last
            try:
                half_sort_val = int(half)
            except ValueError:
                half_sort_val = float('inf') # Sort N/A or non-numeric halves last
            
            # Time sorting: convert MM:SS to total seconds
            try:
                minutes, seconds = map(int, time_str.split(':'))
                time_sort_val = minutes * 60 + seconds
            except ValueError:
                time_sort_val = float('inf') # Should not happen if time_str is always MM:SS
                
            return (half_sort_val, time_sort_val)

        filtered_data_for_csv.sort(key=sort_key)

        Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv_path, 'w', newline='') as csvfile:
            fieldnames = ['Half', 'Time', 'Action', 'Confidence']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(filtered_data_for_csv)
        print(f"Successfully created CSV: {output_csv_path} with {len(filtered_data_for_csv)} rows (sorted).")
    else:
        print(f"No actions met the threshold >= {threshold} (or no valid actions found) in {json_file_path}. CSV not created.")
    print(f"--- Finished processing JSON file: {json_file_path} ---\n")

def main(base_input_dir_str, base_output_dir_str, threshold_val):
    base_input_dir = Path(base_input_dir_str)
    base_output_dir = Path(base_output_dir_str)
    threshold = threshold_val
    
    target_json_filename = "results_spotting.json"

    if not base_input_dir.is_dir():
        print(f"Error: Input directory {base_input_dir} does not exist. Exiting.")
        return

    base_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Starting processing. Input: {base_input_dir}, Output: {base_output_dir}, Threshold: {threshold}")
    print(f"Will look for files named: {target_json_filename}\n")

    json_files_processed_count = 0
    for root, dirs, files in os.walk(base_input_dir):
        if target_json_filename in files:
            json_files_processed_count += 1
            
            input_json_path_str = os.path.join(root, target_json_filename)
            input_json_path = Path(input_json_path_str)

            # Determine output path while preserving structure
            relative_dir_path = Path(root).relative_to(base_input_dir)
            output_csv_dir = base_output_dir / relative_dir_path
            output_csv_path = output_csv_dir / target_json_filename.replace(".json", ".csv")
            
            # print(f"Found '{target_json_filename}' at: {input_json_path}")
            # print(f"Output CSV will be: {output_csv_path}")
            
            process_json_to_csv(input_json_path, output_csv_path, threshold)
    
    if json_files_processed_count == 0:
        print(f"No '{target_json_filename}' files were found in {base_input_dir} or its subdirectories.")
    else:
        print(f"\nProcessed a total of {json_files_processed_count} '{target_json_filename}' files.")
    
    print(f"Processing complete. Output should be in {base_output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process spotting results JSON files to CSV.")
    parser.add_argument(
        "--input_dir", 
        type=str, 
        default="/code/MixSaT/MixSaT_Test_2025/hwatxu3y/results/output_test",
        help="Base directory containing the input JSON files."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="/code/MixSaT/MixSaT_Test_2025/hwatxu3y/results/sub",
        help="Base directory where the output CSV files will be saved."
    )
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=0.6,
        help="Confidence threshold for filtering actions (0.0 to 1.0)."
    )
    args = parser.parse_args()
    
    main(args.input_dir, args.output_dir, args.threshold)
