import os
import shutil
from tqdm import tqdm # Added tqdm import

def copy_specific_soccernet_files(source_base, target_base, files_to_copy_list):
    """
    Copies specific files from a source directory structure to a target directory,
    maintaining the subdirectory structure.

    Args:
        source_base (str): The root directory of the source dataset.
        target_base (str): The root directory where files will be copied.
        files_to_copy_list (list): A list of filenames to copy.
    """
    print(f"Starting file copy operation.")
    print(f"Source base: {source_base}")
    print(f"Target base: {target_base}")
    print(f"Files to copy: {files_to_copy_list}")

    if not os.path.isdir(source_base):
        print(f"Error: Source base directory '{source_base}' does not exist or is not a directory.")
        return

    copied_count = 0
    error_count = 0

    # Collect all directories to process for tqdm
    dir_list = []
    for dirpath, _, _ in os.walk(source_base):
        dir_list.append(dirpath)

    for dirpath in tqdm(dir_list, desc="Processing directories"):
        # Manually get dirnames and filenames for the current dirpath
        # This is a bit less efficient than direct os.walk tuple unpacking but needed for tqdm on dirs
        _, dirnames, filenames = next(os.walk(dirpath))

        # Skip the source_base directory itself if it accidentally contains target files
        # We are interested in files within subdirectories of source_base
        if os.path.samefile(dirpath, source_base) and any(f in filenames for f in files_to_copy_list):
            print(f"Skipping files directly in source_base: {dirpath}")
            # If you intend to copy files from the root of source_base as well, this logic might need adjustment
            # For now, assuming game files are in subdirectories.

        for file_to_copy in files_to_copy_list:
            if file_to_copy in filenames:
                source_file_full_path = os.path.join(dirpath, file_to_copy)
                
                # Determine the relative path from the source_base to the current directory
                relative_dir_path = os.path.relpath(dirpath, source_base)
                
                # Construct the target subdirectory path
                target_subdir_full_path = os.path.join(target_base, relative_dir_path)
                
                # Construct the full target file path
                target_file_full_path = os.path.join(target_subdir_full_path, file_to_copy)
                
                try:
                    # Create the target subdirectory if it doesn't exist
                    if not os.path.exists(target_subdir_full_path):
                        os.makedirs(target_subdir_full_path)
                        print(f"Created directory: {target_subdir_full_path}")
                    
                    # Copy the file
                    shutil.copy2(source_file_full_path, target_file_full_path)
                    print(f"Copied: {source_file_full_path} -> {target_file_full_path}")
                    copied_count += 1
                except Exception as e:
                    print(f"Error copying {source_file_full_path} to {target_file_full_path}: {e}")
                    error_count += 1
    
    print(f"\nFile copy operation completed.")
    print(f"Total files copied: {copied_count}")
    print(f"Total errors: {error_count}")

if __name__ == '__main__':
    # Configuration
    # As per your train.toml, SoccerNet_path is the parent of the individual game folders
    SOURCE_DATASET_ROOT = "/hdda/Datasets/SoccerNet/video" 
    # New location where you want to copy the files, maintaining subfolder structure
    TARGET_DATASET_ROOT_RAW = "~/Datasets/SoccerNet" 
    TARGET_DATASET_ROOT = os.path.expanduser(TARGET_DATASET_ROOT_RAW) # Expand ~
    
    FILES_TO_COPY_CONFIG = [
        "1_baidu_soccer_embeddings.npy",
        "2_baidu_soccer_embeddings.npy",
        "Labels-v2.json",
        "Labels-v3.json" # This file might not exist in all folders; script handles missing files gracefully
    ]

    # Ensure target root directory itself exists or can be created if it's the first level
    if not os.path.exists(TARGET_DATASET_ROOT):
        try:
            os.makedirs(TARGET_DATASET_ROOT)
            print(f"Created target base directory: {TARGET_DATASET_ROOT}")
        except Exception as e:
            print(f"Error creating target base directory {TARGET_DATASET_ROOT}: {e}")
            print("Please ensure you have permissions to create this directory, or create it manually.")
            exit(1)
            
    copy_specific_soccernet_files(SOURCE_DATASET_ROOT, TARGET_DATASET_ROOT, FILES_TO_COPY_CONFIG)
