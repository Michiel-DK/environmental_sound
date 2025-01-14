import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath('../input/'))
from utils import ESC50

train_splits = [1,2,3,4]
test_split = 5

shared_params = {'csv_path': 'audio_data/esc50.csv',
                 'wav_dir': 'audio_data',
                 'dest_dir': 'audio_data/16000',
                 'audio_rate': 16000,
                 'only_ESC10': True,
                 'pad': 0,
                 'normalize': True}

train_gen = ESC50(folds=train_splits,
                  randomize=True,
                  strongAugment=True,
                  random_crop=True,
                  inputLength=2,
                  mix=True,
                  **shared_params).batch_gen(16)

if __name__ == '__main__':
    X, Y = next(train_gen)
    print(X.shape, Y.shape)
    import ipdb; ipdb.set_trace()