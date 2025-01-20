#!/bin/bash

# Copy the tar file
echo "Copying tar file..."
cp /content/drive/MyDrive/environmental_sound/44100_npy.tar . || { echo "File copy failed"; exit 1; }

# Extract the tar file
echo "Extracting tar file..."
tar -xvf 44100_npy.tar || { echo "Extraction failed"; exit 1; }

# Remove specific lines from requirements.txt
echo "Updating requirements.txt..."
sed -i '/-e git+ssh:\/\/git@github.com\/Michiel-DK\/environmental_sound.git@ee1fd2714a59ee6120a90f677157b3d5f13fc00f#egg=environmental_sound/d' requirements.txt
sed -i '/bunch==1.0.1/d' requirements.txt
sed -i '/tensorflow-macos==2.16.2/d' requirements.txt

# Install dependencies from requirements.txt
echo "Installing dependencies from requirements.txt..."
pip install --upgrade --force-reinstall -r requirements.txt || { echo "Dependency installation failed"; exit 1; }

# Install specific version of bunch
echo "Installing bunch==1.0.0..."
pip install bunch==1.0.0 || { echo "Installation of bunch==1.0.0 failed"; exit 1; }

# Install the current directory as a package
echo "Installing the current directory as a package..."
pip install -e . || { echo "Installation of the current directory failed"; exit 1; }

echo "Script completed successfully."
