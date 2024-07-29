import os
import re
import shutil
from pathlib import Path

def reorganize_folders(root_path):
    try:
        root = Path(root_path).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"The directory {root} does not exist.")
        
        print(f"Starting folder reorganization in {root}")
        
        # Regular expression to match date pattern
        date_pattern = re.compile(r'(\d{8})_')
        
        for item in root.iterdir():
            if item.is_dir():
                print(f"Processing directory: {item.name}")
                # Check if it's already a subject folder
                if item.name.startswith('sub-'):
                    subject_folder = item
                else:
                    # Extract subject number from folder name
                    subject_match = re.search(r'sub-(\d+)', item.name)
                    if subject_match:
                        subject_num = subject_match.group(1)
                        subject_folder = root / f'sub-{subject_num}'
                        subject_folder.mkdir(exist_ok=True)
                        print(f"Created subject folder: {subject_folder}")
                    else:
                        print(f"Skipping {item.name}: Unable to determine subject number")
                        continue  # Skip if we can't determine the subject number

                raw_data_folder = subject_folder / 'raw_data'
                raw_data_folder.mkdir(exist_ok=True)
                print(f"Created raw_data folder: {raw_data_folder}")

                # Move contents to raw_data folder and rename
                for subitem in item.iterdir():
                    try:
                        if subitem.is_dir():
                            date_match = date_pattern.search(subitem.name)
                            if date_match:
                                date = date_match.group(1)
                            else:
                                # If date not in folder name, try to get it from files inside
                                for file in subitem.iterdir():
                                    date_match = date_pattern.search(file.name)
                                    if date_match:
                                        date = date_match.group(1)
                                        break
                                else:
                                    print(f"Skipping {subitem.name}: Unable to determine date")
                                    continue  # Skip if we can't determine the date

                            new_folder_name = f"{date}_sub-{subject_folder.name[4:]}_{subitem.name.split('_')[-1]}"
                            new_folder_path = raw_data_folder / new_folder_name
                            
                            # Check if the new folder already exists
                            if new_folder_path.exists():
                                print(f"Folder {new_folder_path} already exists. Moving contents.")
                                # If it exists, we'll move the contents instead of the whole folder
                                for file in subitem.iterdir():
                                    new_file_name = f"{date}_sub-{subject_folder.name[4:]}_{subitem.name.split('_')[-1]}_{file.name.split('_')[-1]}"
                                    new_file_path = new_folder_path / new_file_name
                                    if not new_file_path.exists():
                                        shutil.move(str(file), str(new_file_path))
                                        print(f"Moved file: {file} to {new_file_path}")
                            else:
                                # If it doesn't exist, we can safely move the whole folder
                                shutil.move(str(subitem), str(new_folder_path))
                                print(f"Moved folder: {subitem} to {new_folder_path}")

                        elif subitem.is_file():
                            # Move standalone files directly to raw_data folder
                            shutil.move(str(subitem), str(raw_data_folder / subitem.name))
                            print(f"Moved file: {subitem} to {raw_data_folder / subitem.name}")
                    except Exception as e:
                        print(f"Error processing {subitem}: {str(e)}")

                # Remove original folder if it's empty and not the subject folder
                if item != subject_folder and not any(item.iterdir()):
                    item.rmdir()
                    print(f"Removed empty folder: {item}")

        print("Folder reorganization complete!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

import os
import h5py
from pathlib import Path

def inspect_h5_files(root_path):
    root = Path(root_path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"The directory {root} does not exist.")

    h5_files = sorted(list(root.glob('**/*.h5')))
    
    if not h5_files:
        print("No .h5 files found in the specified directory and its subdirectories.")
        return

    for file_path in h5_files:
        print(f"\nInspecting file: {file_path}")
        try:
            with h5py.File(file_path, 'r') as f:
                print("File attributes:")
                for key, value in f.attrs.items():
                    print(f"  {key}: {value}")
                
                print("\nDatasets:")
                for name, item in f.items():
                    if isinstance(item, h5py.Dataset):
                        print(f"  Dataset: {name}")
                        print(f"    Shape: {item.shape}")
                        print(f"    Type: {item.dtype}")
                        print("    Attributes:")
                        for key, value in item.attrs.items():
                            print(f"      {key}: {value}")
                    elif isinstance(item, h5py.Group):
                        print(f"  Group: {name}")
                        print("    Attributes:")
                        for key, value in item.attrs.items():
                            print(f"      {key}: {value}")
        except Exception as e:
            print(f"Error reading file {file_path}: {str(e)}")
        
        user_input = input("\nPress Enter to continue to the next file, or type 'q' to quit: ")
        if user_input.lower() == 'q':
            break

    print("Inspection complete.")

import os
import json
import h5py
from pathlib import Path

import os
import json
import h5py
from pathlib import Path

def find_matching_files(root):
    h5_files = list(root.glob('**/*.h5'))
    json_files = list(root.glob('**/*.json'))
    return h5_files, json_files

def add_config_to_h5(root_path, overwrite_mode='ask'):
    root = Path(root_path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"The directory {root} does not exist.")

    print(f"Searching for files in {root}")
    h5_files, json_files = find_matching_files(root)

    print(f"Found {len(h5_files)} .h5 files and {len(json_files)} config.json files")

    if not h5_files:
        print("No .h5 files found. Please check the directory path.")
        return

    if not json_files:
        print("No config.json files found. Skipping merging process.")
        return

    for h5_path in h5_files:
        print(f"\nProcessing: {h5_path}")
        
        # Find matching config file
        h5_name = h5_path.stem  # Use stem instead of splitting
        matching_json = next((j for j in json_files if h5_name in j.stem), None)

        if matching_json:
            print(f"Using config: {matching_json}")

            try:
                # Read JSON config
                with open(matching_json, 'r') as json_file:
                    config = json.load(json_file)

                # Add config to H5 file attributes
                with h5py.File(h5_path, 'r+') as h5_file:
                    for key, value in config.items():
                        if key in h5_file.attrs:
                            if overwrite_mode == 'ask':
                                choice = input(f"Attribute '{key}' already exists. Overwrite? (y/n/a for all): ").lower()
                                if choice == 'a':
                                    overwrite_mode = 'overwrite'
                                elif choice != 'y':
                                    continue
                            elif overwrite_mode == 'skip':
                                continue
                            elif overwrite_mode == 'append':
                                h5_file.attrs[key] = str(h5_file.attrs[key]) + '; ' + str(value)
                                continue

                        h5_file.attrs[key] = str(value)  # Convert all values to strings for consistency

                print(f"Successfully added config to {h5_path}")
                print("Removing config file...")
                os.remove(matching_json)
            except Exception as e:
                print(f"Error processing {h5_path}: {str(e)}")

        else:
            print(f"No matching config file found for {h5_path}")

        # user_input = input("\nPress Enter to continue to the next file, or type 'q' to quit: ")
        # if user_input.lower() == 'q':
        #     break

    print("Processing complete.")


import os
import h5py
from pathlib import Path

def rename_dataset(root_path):
    root = Path(root_path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"The directory {root} does not exist.")

    etohbath_folder = root / "etoh_bath_experiments"
    if not etohbath_folder.exists():
        raise FileNotFoundError(f"The etoh_bath_experiments folder does not exist in {root}")

    for subdir in etohbath_folder.iterdir():
        if subdir.is_dir():
            h5_files = sorted(list(subdir.glob("*.h5")))
            
            for h5_path in h5_files:
                print(f"\nProcessing: {h5_path}")
                
                try:
                    with h5py.File(h5_path, 'r+') as h5_file:
                        if 'valve_events' in h5_file:
                            h5_file['events'] = h5_file['valve_events']
                            del h5_file['valve_events']
                            print(f"Successfully renamed 'valve_events' to 'events' in {h5_path}")
                        else:
                            print(f"'valve_events' dataset not found in {h5_path}")
                
                except Exception as e:
                    print(f"Error processing {h5_path}: {str(e)}")
            
            user_input = input("\nPress Enter to continue to the next directory, or type 'q' to quit: ")
            if user_input.lower() == 'q':
                break

    print("Processing complete.")

# Usage
root_path = "~/OneDrive/lgfl/"
root_path = input(f"Enter the root path (default: {root_path}): ") or root_path
# rename_dataset(root_path)

# Usage
# root_path = "~/OneDrive/lgfl"
# root_path = input(f"Enter the root path (default: {root_path}): ") or root_path
# overwrite_mode = input("Choose overwrite mode (ask/skip/overwrite/append): ").lower()
# add_config_to_h5(root_path, "overwrite")
# # Usage
# root_path = "~/OneDrive/lgfl"
inspect_h5_files(root_path)
# # Usage
# root_path = "~/OneDrive/lgfl"
# reorganize_folders(root_path)