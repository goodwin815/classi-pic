import os
import subprocess
import tensorflow as tf
from flask import Flask, request, jsonify

# Function to download the dataset
def download_dataset(dataset, data_dir):
    print(dataset, data_dir)
    if not os.path.exists(data_dir):
        # Create the directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        print(f"Directory {data_dir} created.")
        
        # Download the dataset using Kaggle API
        print("Downloading dataset...")
        subprocess.run(['kaggle', 'datasets', 'download', '-d', dataset, '-p', data_dir, '--unzip'])
        print("Download complete.")
    else:
        print(f"Dataset already exists at {data_dir}.")
