#!/bin/bash

# Set the directory where your output files are located
# Replace this with the actual path to your data directory
DATA_DIR="$HOME/OneDrive/lgfl/rawdata"

# Find and remove all files ending with _output.h5
find "$DATA_DIR" -type f -name "*_output.h5" -delete

echo "Cleanup complete. All *_output.h5 files have been removed."
