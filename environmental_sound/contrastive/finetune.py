import numpy as np
import pytorch_lightning as pl
import torch
from environmental_sound.contrastive.audio_processing import random_crop, random_mask, random_multiply
from environmental_sound.contrastive.audio_dataset_v2 import extract_log_mel_spectrogram, random_mask_proportional

from torch.optim.lr_scheduler import ReduceLROnPlateau



class AudioDatasetSupervised(torch.utils.data.Dataset):
    def __init__(self, data, max_len=100, augment=True):
        self.data = data
        self.max_len = max_len
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        npy_path = self.data[idx][0]
        label = self.data[idx][1]

        x = np.load(npy_path)

        x = random_crop(x, crop_size=self.max_len)

        if self.augment:
            x = random_mask(x)
            x = random_multiply(x)

        x = torch.tensor(x, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long)

        return x, label
    
class ContrastiveAudioDatasetSupervised(torch.utils.data.Dataset):
    """
    Dataset for contrastive learning with audio data and optional labels.

    Args:
        data (list): List of tuples where each tuple contains:
                     (file_path (str), label (int/str)).
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
            tuple: If `augment=True`, returns a tuple of augmented views (x1, x2, label).
                   If `augment=False`, returns the original spectrogram and label.
        """
        file_path, label = self.data[idx]
        waveform = np.load(file_path)

        # Preprocess waveform to log mel spectrogram
        spectrogram = extract_log_mel_spectrogram(waveform)

        if self.augment:
            # Apply augmentations (e.g., masking, cropping)
            spectrogram = random_mask_proportional(spectrogram, mask_ratio=0.15)

            # Create two augmented versions of the spectrogram
            x1 = random_crop(spectrogram, crop_size=self.crop_size)
            x2 = random_crop(spectrogram, crop_size=self.crop_size)

            return (
                torch.tensor(x1, dtype=torch.float),
                torch.tensor(x2, dtype=torch.float),
                label,
            )
        else:
            # For supervised fine-tuning, return a single view and label
            spectrogram = random_crop(spectrogram, crop_size=self.crop_size)
            return torch.tensor(spectrogram, dtype=torch.float), label

    
    
class DecayLearningRate(pl.Callback):
    def __init__(self):
        self.old_lrs = []

    def on_train_start(self, trainer, pl_module):
        # track the initial learning rates
        for opt_idx, optimizer in enumerate(trainer.optimizers):
            group = []
            for param_group in optimizer.param_groups:
                group.append(param_group["lr"])
            self.old_lrs.append(group)

    def on_train_epoch_end(self, trainer, pl_module):
        for opt_idx, optimizer in enumerate(trainer.optimizers):
            old_lr_group = self.old_lrs[opt_idx]
            new_lr_group = []
            for p_idx, param_group in enumerate(optimizer.param_groups):
                old_lr = old_lr_group[p_idx]
                new_lr = old_lr * 0.97
                new_lr_group.append(new_lr)
                param_group["lr"] = new_lr
            self.old_lrs[opt_idx] = new_lr_group
            
class ReduceLROnPlateauCallback(pl.Callback):
    def __init__(self, monitor='val_loss', mode='min', factor=0.5, patience=5, min_lr=1e-6, verbose=True):
        """
        Args:
            monitor (str): The metric to monitor (e.g., 'val_loss').
            mode (str): 'min' to decrease when the metric stops decreasing, 'max' for the opposite.
            factor (float): Factor by which the learning rate will be reduced (new_lr = lr * factor).
            patience (int): Number of epochs to wait before reducing LR after no improvement.
            min_lr (float): Minimum learning rate allowed.
            verbose (bool): Whether to log LR changes.
        """
        self.monitor = monitor
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose
        self.schedulers = []

    def on_train_start(self, trainer, pl_module):
        """Set up ReduceLROnPlateau schedulers for all optimizers."""
        for optimizer in trainer.optimizers:
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode=self.mode,
                factor=self.factor,
                patience=self.patience,
                min_lr=self.min_lr,
                verbose=self.verbose
            )
            self.schedulers.append(scheduler)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Step the scheduler based on the monitored metric."""
        # Retrieve the monitored metric
        metrics = trainer.callback_metrics
        if self.monitor not in metrics:
            raise ValueError(f"Metric '{self.monitor}' not found in trainer's callback metrics.")

        metric_value = metrics[self.monitor].item()

       # Update each scheduler with the metric value
        for scheduler, optimizer in zip(self.schedulers, trainer.optimizers):
            old_lrs = [group['lr'] for group in optimizer.param_groups]
            scheduler.step(metric_value)  # Update the scheduler
            new_lrs = [group['lr'] for group in optimizer.param_groups]

            # Print if the learning rate decreased
            for i, (old_lr, new_lr) in enumerate(zip(old_lrs, new_lrs)):
                if new_lr < old_lr:
                    print(f"Learning rate decreased in optimizer {optimizer}: "
                          f"Group {i} | Old LR: {old_lr:.6f} -> New LR: {new_lr:.6f}")
    
    