# Utility functions for the MixSaT project
import os
import zipfile
import json
import numpy as np
import toml # Added toml import

def zipResults(zip_path, target_dir, filename="results_spotting.json"):
    """Zips files in the target directory that match the given filename."""
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    zipobj = zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED)
    rootlen = len(target_dir) + 1
    for base, dirs, files in os.walk(target_dir):
        for file in files:
            if file == filename:
                fn = os.path.join(base, file)
                zipobj.write(fn, fn[rootlen:])
    zipobj.close()

def get_spot_from_NMS(Input, window=60, thresh=0.0):
    """Extracts spots using Non-Maximum Suppression (NMS)."""
    detections_tmp = np.copy(Input)
    indexes = []
    MaxValues = []
    while(np.max(detections_tmp) >= thresh):
        max_value = np.max(detections_tmp)
        max_index = np.argmax(detections_tmp)
        MaxValues.append(max_value)
        indexes.append(max_index)
        nms_from = int(np.maximum(-(window/2)+max_index, 0))
        nms_to = int(np.minimum(max_index+int(window/2), len(detections_tmp)))
        detections_tmp[nms_from:nms_to] = -1
    return np.transpose([indexes, MaxValues])

def format_results_to_json(game_ID, timestamp_long_half_1, timestamp_long_half_2, 
                           output_results_path, framerate, NMS_window, NMS_threshold, 
                           INVERSE_EVENT_DICTIONARY_V2):
    """Formats spotting results into JSON and saves to a file."""
    json_data = dict()
    json_data["UrlLocal"] = game_ID
    json_data["predictions"] = list()

    for half, timestamp in enumerate([timestamp_long_half_1, timestamp_long_half_2]):
        for l in range(17): # Assuming 17 event types, excluding null class
            spots = get_spot_from_NMS(
                timestamp[:, l], 
                window=NMS_window*framerate, 
                thresh=NMS_threshold
            )
            for spot in spots:
                frame_index = int(spot[0])
                confidence = spot[1]
                seconds = int((frame_index // framerate) % 60)
                minutes = int((frame_index // framerate) // 60)

                prediction_data = dict()
                prediction_data["gameTime"] = f"{half+1} - {minutes}:{seconds}"
                prediction_data["label"] = INVERSE_EVENT_DICTIONARY_V2[l]
                prediction_data["position"] = str(int((frame_index / framerate) * 1000))
                prediction_data["half"] = str(half+1)
                prediction_data["confidence"] = str(confidence)
                json_data["predictions"].append(prediction_data)

    game_output_dir = os.path.join(output_results_path, game_ID)
    os.makedirs(game_output_dir, exist_ok=True)
    with open(os.path.join(game_output_dir, 'results_spotting.json'), 'w') as output_file:
        json.dump(json_data, output_file, indent=4)
    return 0

def load_and_flatten_toml_config(config_path):
    try:
        with open(config_path, 'r') as f:
            nested_config = toml.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        raise
    except Exception as e:
        print(f"Error loading configuration file '{config_path}': {e}")
        raise

    flat_config = {}
    for section_name, section_content in nested_config.items():
        if isinstance(section_content, dict):
            for key, value in section_content.items():
                if key in flat_config:
                    print(f"Warning: Duplicate key '{key}' found in TOML. Value from section '{section_name}' (value: {value}) will overwrite previous value: {flat_config[key]}.")
                flat_config[key] = value
        else:
            if section_name in flat_config:
                 print(f"Warning: Duplicate key '{section_name}' found at top level of TOML. Value will overwrite previous.")
            flat_config[section_name] = section_content
    return flat_config
