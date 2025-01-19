import json
from glob import glob

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from environmental_sound.contrastive.audio_processing import random_crop, random_mask, random_multiply
from environmental_sound.contrastive.encoder import AudioClassifier


class AudioDatasetSupervised(torch.utils.data.Dataset):
    def __init__(self, data, max_len=512, augment=True):
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
    
    