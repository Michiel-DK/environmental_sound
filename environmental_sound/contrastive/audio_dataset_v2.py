import numpy as np
import random
import torch
from torch.utils.data import Dataset
import tensorflow as tf

# -------------------------------
# Preprocessing and Augmentation
# -------------------------------

def extract_log_mel_spectrogram(
    waveform,
    sample_rate=44100,
    frame_length=2048,  # ~46 ms window
    frame_step=1024,    # ~23 ms step for 50% overlap
    fft_length=2048,    # Matches frame_length
    n_mels=64,          # Retain 64 mel bands
    fmin=60.0,          # Keep the same
    fmax=22050.0,       # Extend to Nyquist frequency
):
    """
    Extracts log mel spectrogram from a raw waveform.

    Args:
        waveform (ndarray): Input audio waveform as a 1D numpy array.
        sample_rate (int): Sampling rate of the audio (in Hz). Default is 16,000 Hz.
        frame_length (int): Length of each frame for STFT (in samples). Default is 400.
        frame_step (int): Step size between consecutive frames (in samples). Default is 160.
        fft_length (int): Number of FFT points for STFT. Default is 1024.
        n_mels (int): Number of mel bands to generate. Default is 64.
        fmin (float): Minimum frequency for mel filter bank (in Hz). Default is 60 Hz.
        fmax (float): Maximum frequency for mel filter bank (in Hz). Default is 7800 Hz.

    Returns:
        ndarray: Log mel spectrogram of shape `(T, n_mels)`, where `T` is the number of frames.
    """
        
    stfts = tf.signal.stft(
        waveform, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length
    )
    spectrograms = tf.abs(stfts)
    num_spectrogram_bins = stfts.shape[-1]
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        n_mels, num_spectrogram_bins, sample_rate, fmin, fmax
    )
    mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(
        spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:])
    )
    mel_spectrograms = tf.clip_by_value(mel_spectrograms, 1e-5, 1e8)
    log_mel_spectrograms = tf.math.log(mel_spectrograms)
    return log_mel_spectrograms


def random_mask(data, rate_start=0.2, rate_seq=0.3):
    """
    Masks random regions of the spectrogram to simulate missing data or noise.

    Args:
        data (ndarray): Input spectrogram of shape `(T, n_mels)`.
        rate_start (float): Probability of masking the start of a frame. Default is 0.1.
        rate_seq (float): Probability of continuing a masking sequence. Default is 0.2.

    Returns:
        ndarray: Spectrogram with masked regions.
    """
    # new_data = data.copy()
    # mean = new_data.mean()
    new_data = tf.identity(data)
    mean = tf.reduce_mean(new_data)  # Calculate the mean of the data
    prev_zero = False
    # Iterate over the rows of the tensor
    for i in range(new_data.shape[0]):
        # Use Python's random.random for stochastic masking
        if random.random() < rate_start or (prev_zero and random.random() < rate_seq):
            prev_zero = True
            # Mask the row with the mean
            new_data = tf.tensor_scatter_nd_update(
                new_data,
                indices=[[i]],
                updates=[tf.fill(new_data[i, :].shape, mean)],
            )
        else:
            prev_zero = False
    return new_data


def random_mask_proportional(data, mask_ratio=0.15):
    """
    Masks a proportion of the spectrogram to simulate missing data or noise.

    Args:
        data (ndarray): Input spectrogram of shape `(T, n_mels)`.
        mask_ratio (float): Fraction of time frames to mask. Default is 0.15 (15%).

    Returns:
        ndarray: Spectrogram with masked regions.
    """
    new_data = data.copy()
    mean = new_data.mean()
    T = new_data.shape[0]
    num_masked_frames = int(mask_ratio * T)
    masked_indices = random.sample(range(T), num_masked_frames)

    for idx in masked_indices:
        new_data[idx, :] = mean  # Mask the entire frame

    return new_data



def random_multiply(data):
    """
    Scales the spectrogram intensity with a random factor.

    Args:
        data (ndarray): Input spectrogram of shape `(T, n_mels)`.

    Returns:
        ndarray: Scaled spectrogram.
    """
    return data * (0.9 + random.random() / 5.0)


def random_crop(data, crop_size=128):
    """
    Extracts a random crop from the spectrogram.

    Args:
        data (ndarray): Input spectrogram of shape `(T, n_mels)`.
        crop_size (int): Number of frames to crop. Default is 128.

    Returns:
        ndarray: Cropped spectrogram of shape `(crop_size, n_mels)`.
    """
    start = random.randint(0, data.shape[0] - crop_size)
    return data[start : start + crop_size, :]


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
            torch.tensor(x1.numpy(), dtype=torch.float32),
            torch.tensor(x2.numpy(), dtype=torch.float32),
        )

