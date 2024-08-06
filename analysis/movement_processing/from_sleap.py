import os
import shutil
from pathlib import Path

def create_directory_structure_and_move_analysis_files(base_path, analysis_path):
    # Create the derivatives directory
    derivatives_path = Path(base_path) / "derivatives"
    derivatives_path.mkdir(exist_ok=True)

    # Create empty directory structure in derivatives
    for item in Path(base_path).iterdir():
        if item.is_dir() and "raw" in item.name:
            print(f"Processing {item.name}")
            for sub_item in item.iterdir():
                if sub_item.is_dir() and sub_item.name.startswith("sub-"):
                    print(f"Creating directory structure for {sub_item.name}")
                    for directory in sub_item.iterdir():
                        new_dir = derivatives_path / sub_item.name / directory.name
                        print(f"Creating: {new_dir}")
                        new_dir.mkdir(parents=True, exist_ok=True)

    # Move analysis files to the appropriate directories
    for analysis_file in Path(analysis_path).glob("*.analysis.h5"):
        # Extract subject number and date from filename
        parts = analysis_file.stem.split("_")
        subject_num = parts[-2]
        date = parts[-3]

        # Construct the path for the destination directory
        dest_dir = derivatives_path / f"{subject_num}" / f"{date}_{subject_num}_OnePortEtohExperiment"
        print("Desitination directory", dest_dir)

        if dest_dir.exists():
            shutil.move(str(analysis_file), str(dest_dir / analysis_file.name))
            print(f"Moved {analysis_file.name} to {dest_dir}")
        else:
            print(f"Destination directory not found for {analysis_file.name}")

    print("Directory structure creation and analysis file moving complete.")


# Usage
base_path = "/home/jmdelahanty/OneDrive/lgfl"
analysis_path = "/home/jmdelahanty/ahrens_data/lgfl/oneportetoh_sleap/"
create_directory_structure_and_move_analysis_files(base_path, analysis_path)