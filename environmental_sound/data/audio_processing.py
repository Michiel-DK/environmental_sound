import random

import librosa
import numpy as np
from tqdm import tqdm
import os

input_length = 22050 * 4

n_mels = 64


def pre_process_audio_mel_t(audio, sample_rate=44000):
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=n_mels)
    mel_db = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40

    return mel_db.T


def load_audio_file(file_path, input_length=input_length):
    try:
        data = librosa.core.load(file_path, sr=22050)[0]  # , sr=16000
    except ZeroDivisionError:
        data = []

    if len(data) > input_length:
        
        max_offset = len(data) - input_length

        offset = np.random.randint(max_offset)

        data = data[offset : (input_length + offset)]

    else:
        if input_length > len(data):
            print(f'{file_path} - smaller')

            max_offset = input_length - len(data)

            offset = np.random.randint(max_offset)
        else:
            offset = 0

        data = np.pad(data, (offset, input_length - len(data) - offset), "constant")

    #data = pre_process_audio_mel_t(data)
    return data

def load_audio_file_strict(file_path, input_length=input_length):
    try:
        data = librosa.core.load(file_path, sr=22050)[0]  # , sr=16000
    except ZeroDivisionError:
        data = []

    if len(data) != input_length:
        
        pass

    else:

        return data


def random_crop(data, crop_size=128):
    start = int(random.random() * (data.shape[0] - crop_size))
    return data[start : (start + crop_size), :]


def random_mask(data, rate_start=0.1, rate_seq=0.2):
    new_data = data.copy()
    mean = new_data.mean()
    prev_zero = False
    for i in range(new_data.shape[0]):
        if random.random() < rate_start or (
            prev_zero and random.random() < rate_seq
        ):
            prev_zero = True
            new_data[i, :] = mean
        else:
            prev_zero = False

    return new_data


def random_multiply(data):
    new_data = data.copy()
    return new_data * (0.9 + random.random() / 5.)


def save(input_path, output_path):
    data = load_audio_file_strict(input_path)
    np.save(output_path, data)
    return True


if __name__ == "__main__":

    input_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'audio_data/8k/22050/')
    output_data_path = os.path.join(os.path.dirname(os.path.dirname(input_data_path)), '22050_npy_nopre/')
    
    files = os.listdir(input_data_path)
    
    import pandas as pd
    
    df = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(input_data_path)), 'labels.csv'))
    df_filtered = df[df['check']==1]
    
    filtered_list = list(df_filtered.filename.to_list())
    
    filtered_files = list(set(files) & set(filtered_list))
        
    for i, file in tqdm(enumerate(filtered_files), total=len(filtered_files)):
        
        input_file = os.path.join(input_data_path, file)
        output_file = os.path.join(output_data_path, file.replace('.wav', '.npy'))
        
        save(input_file, output_file)