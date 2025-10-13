#!/bin/bash

# Create the data directory if it doesn't exist
mkdir -p DATA/object_study_raw

# Download the data files from Google Drive
echo "Downloading data files from Google Drive..."
gdown --folder https://drive.google.com/drive/folders/1PBby4Sja-j1e8alLoX8eFvUw0BH-LCcd -O DATA/obj_study_raw

# Download the masks from Google Drive
echo "Downloading masks from Google Drive..."
gdown --folder https://drive.google.com/drive/folders/1yelMdtds9VTstfxDjqux0K6vlZeAJyFQ -O DATA/obj_study_proc

# Check if download was successful
if [ $? -eq 0 ]; then
    echo "Data downloaded successfully to DATA/object_study_raw/"
else
    echo "Error downloading data. Please make sure you have gdown installed and have access to the Google Drive folder."
    echo "You can install gdown using: pip install gdown"
fi 
