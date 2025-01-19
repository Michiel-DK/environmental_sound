import os
import tarfile
from google.cloud import storage

def check_and_setup_directory(root_path, output_data_path, bucket_name, tar_blob_name):
    """
    Checks if a directory exists, downloads a .tar file from a Google Cloud Storage bucket, and extracts it if needed.

    Parameters:
        root_path (str): The root directory path.
        output_data_path (str): The directory path to check.
        bucket_name (str): The name of the GCS bucket.
        tar_blob_name (str): The blob name of the .tar file in the bucket.
    """
    if not os.path.exists(output_data_path):
        print(f"Directory {output_data_path} does not exist. Creating it...")
        os.makedirs(output_data_path, exist_ok=True)

        # Initialize Google Cloud Storage client
        storage_client = storage.Client(project='contrastive-learning-440810')
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(tar_blob_name)

        # Set local tar file path
        local_tar_path = os.path.join(root_path, tar_blob_name)

        # Download the .tar file
        print(f"Downloading {tar_blob_name} from bucket {bucket_name}...")
        blob.download_to_filename(local_tar_path)
        print(f"Downloaded {tar_blob_name} to {local_tar_path}.")

        # Extract the .tar file
        print(f"Extracting {local_tar_path}...")
        with tarfile.open(local_tar_path, 'r') as tar:
            tar.extractall(path=root_path)
        print(f"Extracted {local_tar_path} to {root_path}.")

        # Cleanup: Remove the downloaded .tar file
        os.remove(local_tar_path)
        print(f"Removed temporary file {local_tar_path}.")
    else:
        print(f"Directory {output_data_path} already exists.")
