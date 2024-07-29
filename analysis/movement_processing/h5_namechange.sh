#!/bin/bash

# Set the base directory
BASE_DIR="$HOME/OneDrive/lgfl/rawdata"

# Function to rename files
rename_files() {
    local dir="$1"
    for file in "$dir"/*_timestamps.h5; do
        if [ -f "$file" ]; then
            new_name="${file%_timestamps.h5}.h5"
            mv "$file" "$new_name"
            echo "Renamed: $file -> $new_name"
        fi
    done
}

# Main loop to traverse directories
find "$BASE_DIR" -type d -name "*OnePort*" | while read -r dir; do
    echo "Processing directory: $dir"
    rename_files "$dir"
done

echo "File renaming complete."
