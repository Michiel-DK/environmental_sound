import numpy as np
import torch
from torch.utils.data import Dataset

from environmental_sound.contrastive.dataset_utils import *

class ContrastiveAudioDatasetUnsupervised(Dataset):
    """
    Dataset for contrastive learning with audio data.

    Args:
        data (list): List of file paths to the audio data (e.g., .npy files).
        augment (bool): Whether to apply augmentations. Default is True.
        seg_length (int): Length of the audio segments (in samples). Default is 16,000.
        crop_size (int): Number of frames for cropping spectrograms. Default is 128.
    """

    def __init__(self, data, augment=True, seg_length=44100, crop_size=128):
        self.data = data
        self.augment = augment
        self.seg_length = seg_length
        self.crop_size = crop_size
        self.error_count = 0  # Initialize error counter

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Total number of samples.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves and processes a single sample.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: Two augmented views of the spectrogram as tensors `(x1, x2)`.
        """
        npy_path = self.data[idx]
        try:
            waveform = np.load(npy_path)
        except ValueError as e:
            if "allow_pickle" in str(e):
                # Increment error counter and skip this sample
                self.error_count += 1
                print(f"Error loading {npy_path}: {e}. Skipping this file.")
                return None  # Return None to indicate a problematic file

        
        # Preprocess waveform to log mel spectrogram
        spectrogram = extract_log_mel_spectrogram(waveform)

        if self.augment:
            # Apply augmentations
            spectrogram = random_mask_proportional(spectrogram, mask_ratio=0.15)
            x1 = random_crop(spectrogram, crop_size=self.crop_size)
            x2 = random_crop(spectrogram, crop_size=self.crop_size)

            # Add diversity with additional augmentations => to check with same tranformations
            x1 = random_time_mask(x1, mask_ratio=0.1)
            x1 = random_frequency_mask(x1, mask_ratio=0.1)
            
            x2 = random_time_mask(x2, mask_ratio=0.1)
            x2 = add_random_noise(x2, noise_level=0.05)
        
        else:
            x1 = center_crop(spectrogram, crop_size=self.crop_size)
            x2 = center_crop(spectrogram, crop_size=self.crop_size)

        return (
            torch.tensor(x1, dtype=torch.float32),
            torch.tensor(x2, dtype=torch.float32),
        )
        
class ContrastiveAudioDatasetSupervised(torch.utils.data.Dataset):
    def __init__(self, data, augment=True, seg_length=44100, crop_size=128):
        self.data = data
        self.augment = augment
        self.seg_length = seg_length
        self.crop_size = crop_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path, label = self.data[idx]
        waveform = np.load(file_path)

        # Preprocess waveform to log mel spectrogram
        spectrogram = extract_log_mel_spectrogram(waveform)

        if self.augment:
            # Apply augmentations
            spectrogram = random_mask_proportional(spectrogram, mask_ratio=0.15)
            x1 = random_crop(spectrogram, crop_size=self.crop_size)
            x2 = random_crop(spectrogram, crop_size=self.crop_size)

            # Add diversity with additional augmentations
            x1 = random_time_mask(x1, mask_ratio=0.1)
            x1 = random_frequency_mask(x1, mask_ratio=0.1)
            
            x2 = random_time_mask(x2, mask_ratio=0.1)
            x2 = add_random_noise(x2, noise_level=0.05)

            return (
                torch.tensor(x1, dtype=torch.float),
                torch.tensor(x2, dtype=torch.float),
                label,
            )
        else:
            # For supervised fine-tuning, return a single view and label
            spectrogram = random_crop(spectrogram, crop_size=self.crop_size)
            return torch.tensor(spectrogram, dtype=torch.float), label


# Wrapper function for handling None entries during data loading
def collate_fn_with_skip(batch):
    # Filter out None entries
    batch = [item for item in batch if item is not None]
    return torch.utils.data.dataloader.default_collate(batch)