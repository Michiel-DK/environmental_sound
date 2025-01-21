import numpy as np
import random
import torch
from torch.utils.data import Dataset

# -------------------------------
# Preprocessing and Augmentation
# -------------------------------

import numpy as np
import librosa

def extract_log_mel_spectrogram(
    waveform,
    sample_rate=44100,
    frame_length=2048,  # ~46 ms window
    frame_step=1024,    # ~23 ms step for 50% overlap
    fft_length=2048,    # Matches frame_length
    n_mels=64,          # Retain 64 mel bands
    fmin=60.0,          # Minimum frequency for mel filter bank
    fmax=None,          # Extend to Nyquist frequency if None
):
    """
    Extracts log mel spectrogram from a raw waveform using librosa.

    Args:
        waveform (ndarray): Input audio waveform as a 1D numpy array.
        sample_rate (int): Sampling rate of the audio (in Hz).
        frame_length (int): Length of each frame for STFT (in samples).
        frame_step (int): Step size between consecutive frames (in samples).
        fft_length (int): Number of FFT points for STFT.
        n_mels (int): Number of mel bands to generate.
        fmin (float): Minimum frequency for mel filter bank (in Hz).
        fmax (float): Maximum frequency for mel filter bank (in Hz). If None, it defaults to Nyquist frequency.

    Returns:
        ndarray: Log mel spectrogram of shape `(n_mels, T)`, where `T` is the number of frames.
    """
    # Compute the mel spectrogram
    hop_length = frame_step
    mel_spectrogram = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_fft=fft_length,
        hop_length=hop_length,
        win_length=frame_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0,  # Power of 2 corresponds to energy
    )

    # Convert to log scale (dB)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return log_mel_spectrogram



def random_mask(data, rate_start=0.2, rate_seq=0.3):
    """
    Masks random regions of the spectrogram to simulate missing data or noise.

    Args:
        data (ndarray): Input spectrogram of shape `(n_mels, T)`.
        rate_start (float): Probability of masking the start of a frame. Default is 0.2.
        rate_seq (float): Probability of continuing a masking sequence. Default is 0.3.

    Returns:
        ndarray: Spectrogram with masked regions.
    """
    new_data = data.copy()
    mean = new_data.mean()
    prev_zero = False

    for i in range(new_data.shape[1]):  # Iterate over time frames (T)
        if random.random() < rate_start or (prev_zero and random.random() < rate_seq):
            prev_zero = True
            new_data[:, i] = mean  # Mask the entire column (time frame)
        else:
            prev_zero = False

    return new_data


def random_mask_proportional(data, mask_ratio=0.15):
    """
    Masks a proportion of the spectrogram to simulate missing data or noise.

    Args:
        data (ndarray): Input spectrogram of shape `(n_mels, T)`.
        mask_ratio (float): Fraction of time frames to mask. Default is 0.15 (15%).

    Returns:
        ndarray: Spectrogram with masked regions.
    """
    new_data = data.copy()
    mean = new_data.mean()
    T = new_data.shape[1]
    num_masked_frames = int(mask_ratio * T)
    masked_indices = random.sample(range(T), num_masked_frames)

    for idx in masked_indices:
        new_data[:, idx] = mean  # Mask the entire column (time frame)

    return new_data


def random_multiply(data):
    """
    Scales the spectrogram intensity with a random factor.

    Args:
        data (ndarray): Input spectrogram of shape `(n_mels, T)`.

    Returns:
        ndarray: Scaled spectrogram.
    """
    return data * (0.9 + random.random() / 5.0)


def random_crop(data, crop_size=128):
    """
    Extracts a random crop from the spectrogram.

    Args:
        data (ndarray): Input spectrogram of shape `(n_mels, T)`.
        crop_size (int): Number of time frames to crop. Default is 128.

    Returns:
        ndarray: Cropped spectrogram of shape `(n_mels, crop_size)`.
    """
    start = random.randint(0, data.shape[1] - crop_size)
    return data[:, start : start + crop_size]


class ContrastiveAudioDataset(Dataset):
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
        waveform = np.load(npy_path)
        
        # Preprocess waveform to log mel spectrogram
        spectrogram = extract_log_mel_spectrogram(waveform)


        if self.augment:
            spectrogram = random_mask(spectrogram)
            spectrogram = random_multiply(spectrogram)


        x1 = random_crop(spectrogram, crop_size=self.crop_size)
        x2 = random_crop(spectrogram, crop_size=self.crop_size)

        return (
            torch.tensor(x1, dtype=torch.float32),
            torch.tensor(x2, dtype=torch.float32),
        )

class AudioDatasetSupervised(torch.utils.data.Dataset):
    def __init__(self, data, augment=True, crop_size=128):
        self.data = data
        self.augment = augment
        self.crop_size = crop_size
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        npy_path = self.data[idx][0]
        label = self.data[idx][1]

        waveform = np.load(npy_path)
        
        # Preprocess waveform to log mel spectrogram
        spectrogram = extract_log_mel_spectrogram(waveform)

        if self.augment:
            spectrogram = random_mask(spectrogram)
            spectrogram = random_multiply(spectrogram)
            
        x = random_crop(spectrogram, crop_size=self.crop_size)

        x = torch.tensor(x, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return x, label