import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
import librosa
import pandas as pd
from tqdm import tqdm
import tensorflow as tf  # For one-hot encoding; alternatively use PyTorch methods
import cv2  # For resizing images


class MFCCDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        df, 
        data_path='audio_data/44100',  # Default directory path
        batch_size=32, 
        num_workers=4,
        train_prefixes=['1', '2', '3'], 
        val_prefixes=['4'], 
        test_prefixes=['5'],
        num_classes=50,
        resize = False,
        target_size=(224, 224)  # Desired size for each MFCC image (height, width)
    ):
        """
        Args:
            df (pd.DataFrame): DataFrame containing filenames and labels.
            data_path (str): Path to the directory containing audio files. Defaults to 'audio_data/44100'.
            batch_size (int): Batch size for DataLoaders.
            num_workers (int): Number of worker threads for DataLoaders.
            train_prefixes (list): List of filename prefixes for training set.
            val_prefixes (list): List of filename prefixes for validation set.
            test_prefixes (list): List of filename prefixes for test set.
            num_classes (int): Number of output classes for one-hot encoding.
        """
        super().__init__()
        self.df = df
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_prefixes = train_prefixes
        self.val_prefixes = val_prefixes
        self.test_prefixes = test_prefixes
        self.num_classes = num_classes
        self.resize = resize
        self.target_size = target_size  # (height, width)


    def setup(self, stage=None):
        # Filter DataFrame based on filename prefixes for train, val, test splits
        self.train_df = self.df[self.df.iloc[:, 0].apply(
            lambda x: any(str(x).startswith(prefix) for prefix in self.train_prefixes)
        )]
        self.val_df = self.df[self.df.iloc[:, 0].apply(
            lambda x: any(str(x).startswith(prefix) for prefix in self.val_prefixes)
        )]
        self.test_df = self.df[self.df.iloc[:, 0].apply(
            lambda x: any(str(x).startswith(prefix) for prefix in self.test_prefixes)
        )]

        # Process each subset to compute MFCC features and create datasets
        self.train_dataset = self.process_df(self.train_df)
        self.val_dataset = self.process_df(self.val_df)
        self.test_dataset = self.process_df(self.test_df)

    def process_df(self, subset_df):
        """Process a DataFrame subset to compute MFCCs, resize them, and one-hot encode labels."""
        X, y = [], []

        # Desired dimensions for resizing
        target_height, target_width = self.target_size

        for _, row in tqdm(subset_df.iterrows(), total=len(subset_df), desc="Processing"):
            # Extract filename and label from DataFrame row
            filename, label = row.iloc[0], row.iloc[2]
            file_path = os.path.join(self.data_path, filename)
            try:
                sig, sr = librosa.load(file_path, sr=None)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue

            # Extract multiple segments per audio file
            for _ in range(3):
                if len(sig) < sr * 2:
                    continue
                start_idx = np.random.randint(0, len(sig) - (sr * 2))
                sig_segment = sig[start_idx : start_idx + (sr * 2)]
                mfcc = librosa.feature.mfcc(y=sig_segment, sr=sr, n_mfcc=13)

                # Resize MFCC if it does not match the target size
                if self.resize:
                    # cv2.resize expects (width, height)
                    mfcc = cv2.resize(mfcc, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

                X.append(mfcc)
                y.append(label)
        
        # Convert lists to numpy arrays
        X = np.array(X)
        y = np.array(y)
        # Convert lists to arrays
        X = np.array(X)
        y = np.array(y)

        # One-hot encode labels
        y_encoded = tf.keras.utils.to_categorical(y, num_classes=self.num_classes)

        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y_encoded, dtype=torch.float32)

        return TensorDataset(X_tensor, y_tensor)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True
        )

if __name__ == '__main__':
    
    labels_df = pd.read_csv('audio_data/esc50.csv')
    
    data_module = MFCCDataModule(df=labels_df, batch_size=32, num_workers=4, resize=False)

    # Setup to prepare datasets and dataloaders
    data_module.setup()

    # Retrieve dataloaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    
    for x in train_loader:
        x_test = x
        break
    
    for x in val_loader:
        x_val = x
        break
    
    import ipdb;ipdb.set_trace()