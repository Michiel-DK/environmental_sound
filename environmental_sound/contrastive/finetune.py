import numpy as np
import pytorch_lightning as pl
import torch
from environmental_sound.contrastive.audio_processing import random_crop, random_mask, random_multiply

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
    
    