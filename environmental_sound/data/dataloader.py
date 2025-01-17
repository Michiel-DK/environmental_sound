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

from environmental_sound.supervised.transformations import RandomAudio, TimeStretch, MelSpectrogram, SpecAugment, SpectToImage

import albumentations


class MFCCDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        df, 
        data_path='audio_data/44100',  # Default directory path
        batch_size=32, 
        num_workers=4,
        num_classes=50,
        resize=False,
        target_size=(224, 224),
        audio_transform_train=None,
        audio_transform_val_test=None,
        use_mfcc=True,
        train_pct=0.75,
        val_pct=0.15,
        test_pct=0.10,
        sample_subset=3
    ):
        """
        Args:
            df (pd.DataFrame): DataFrame containing filenames and labels.
            data_path (str): Path to the directory containing audio files. 
            batch_size (int): Batch size for DataLoaders.
            num_workers (int): Number of worker threads for DataLoaders.
            train_prefixes (list): List of filename prefixes for training set.
            val_prefixes (list): List of filename prefixes for validation set.
            test_prefixes (list): List of filename prefixes for test set.
            num_classes (int): Number of output classes for one-hot encoding.
            resize (bool): Whether to resize spectrogram/MFCC features.
            target_size (tuple): The desired (height, width) for resizing.
            audio_transform: Albumentations Compose for audio transformations.
            use_mfcc (bool): If True, compute MFCCs from the (optionally) transformed audio. 
                             If False, skip MFCC and use the transform’s output directly.
        """
        super().__init__()
        self.df = df
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = num_classes
        self.resize = resize
        self.target_size = target_size
        self.audio_transform_train = audio_transform_train
        self.audio_transform_val_test = audio_transform_val_test
        self.use_mfcc = use_mfcc
        self.sample_subset = sample_subset
        
        # Save percentage splits
        self.train_pct = train_pct
        self.val_pct = val_pct
        self.test_pct = test_pct
    
    def setup(self, stage=None):
        # Shuffle the entire dataset
        df_shuffled = self.df.sample(frac=1, random_state=42).reset_index(drop=True)
        n = len(df_shuffled)

        # Calculate split indices based on percentages
        train_end = int(n * self.train_pct)
        val_end = train_end + int(n * self.val_pct)

        # Split the dataframe
        self.train_df = df_shuffled.iloc[:train_end]
        self.val_df = df_shuffled.iloc[train_end:val_end]
        self.test_df = df_shuffled.iloc[val_end:]

        # Process each subset to compute features (MFCC or transform's output) and create datasets
        self.train_dataset = self.process_df(self.train_df, transform=self.audio_transform_train)
        self.val_dataset = self.process_df(self.val_df, transform=self.audio_transform_val_test)
        self.test_dataset = self.process_df(self.test_df, transform=self.audio_transform_val_test)

    def process_df(self, subset_df, transform):
        """Process a DataFrame subset to either compute MFCCs or apply the optional transform pipeline."""
        X, y = [], []

        target_height, target_width = self.target_size

        for _, row in tqdm(subset_df.iterrows(), total=len(subset_df), desc="Processing"):
            filename, label = row.iloc[0], row.iloc[2]
            file_path = os.path.join(self.data_path, filename)

            try:
                sig, sr = librosa.load(file_path, sr=None)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue

            # Extract multiple segments per audio file (example: 3 segments)
            for _ in range(self.sample_subset):
                if len(sig) < sr * 2:
                    # Audio too short to reliably segment
                    continue

                start_idx = np.random.randint(0, len(sig) - (sr * 2))
                sig_segment = sig[start_idx : start_idx + (sr * 2)]

                # --- 1) If an audio_transform is given, apply it first ---
                # This could return raw audio or a spectrogram (depending on the transforms).
                if transform is not None:
                    transformed = transform(data=(sig_segment, sr))
                    data_out, sr_out = transformed["data"]
                                        
                else:
                    # No transform, just keep the raw data
                    data_out, sr_out = sig_segment, sr

                # --- 2) If we still want MFCC after transforms (use_mfcc=True), compute it ---
                if self.use_mfcc:
                    # If the transform has already created a spectrogram,
                    #   data_out will be 2D. In that case, skip MFCC or handle it differently.
                    #   For simplicity, let's assume data_out is still raw audio if we want MFCC.
                    if isinstance(data_out, np.ndarray) and data_out.ndim == 1:
                        # data_out is 1D => raw audio, so compute MFCC
                        features = librosa.feature.mfcc(y=data_out, sr=sr_out, n_mfcc=64)
                    else:
                        # data_out might already be a spectrogram
                        features = data_out
                else:
                    # If not computing MFCC, just use the transform’s output directly
                    features = data_out

                # --- 3) Optionally resize (for CNN input, etc.) ---
                if self.resize and features.ndim == 2:
                    # (height, width) in OpenCV => (target_width, target_height)
                    features = cv2.resize(
                        features,
                        (target_width, target_height),
                        interpolation=cv2.INTER_LINEAR
                    )

                X.append(features)
                y.append(label)
                
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
    
    train_transform = albumentations.Compose([
        RandomAudio(seconds=2, p=0.5),    # <-- augmentation
        TimeStretch(p=0.5),              # <-- augmentation
        #MelSpectrogram(parameters={"n_mels": 128, "fmax": 8000, "n_mfcc":13}, p=1.0),  # <-- necessary
        SpecAugment(p=0.5),              # <-- augmentation
        SpectToImage(p=1.0)              # <-- can be seen as final step
    ])

    val_test_transform = albumentations.Compose([
        # No random augmentation here;
        # Just do the minimal steps needed to get the final representation
        #MelSpectrogram(parameters={"n_mels": 128, "fmax": 8000, "n_mfcc":13}, p=1.0),
        SpectToImage(p=1.0)
    ])
    
    data_module = MFCCDataModule(df=labels_df, batch_size=32, audio_transform_train=train_transform,use_mfcc=True, audio_transform_val_test=val_test_transform)
    #data_module = MFCCDataModule(df=labels_df, batch_size=32,use_mfcc=True)
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