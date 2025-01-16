import albumentations
from albumentations.core.transforms_interface import DualTransform, BasicTransform
import random
import numpy as np
import librosa
from environmental_sound.utils import SerializableTransformMixin

class AudioTransform(BasicTransform):
    """Transform for Audio task"""

    @property
    def targets(self):
        return {"data": self.apply}
    
    def update_params(self, params, **kwargs):
        if hasattr(self, "interpolation"):
            params["interpolation"] = self.interpolation
        if hasattr(self, "fill_value"):
            params["fill_value"] = self.fill_value
        return params
    
class TimeStretch(SerializableTransformMixin, AudioTransform):
    """Shifting time axis"""
    def __init__(self, always_apply=False, p=0.5):
        super(TimeStretch, self).__init__(always_apply, p)
    
    def apply(self, data, **params):
        sound, sr = data

        rate = np.random.uniform(0, 2)
        augmented_sound = librosa.effects.time_stretch(sound, rate)

        return augmented_sound, sr
    
class RandomAudio(SerializableTransformMixin, AudioTransform):
    """Shifting time axis"""
    def __init__(self,  seconds=5, always_apply=False, p=0.5):
        super(RandomAudio, self).__init__(always_apply, p)

        self.seconds = seconds
    
    def apply(self, data, **params):
        sound, sr = data

        shift = np.random.randint(len(sound))
        trim_sound = np.roll(sound, shift)

        min_samples = int(sr * self.seconds)

        if len(trim_sound) < min_samples:
            padding = min_samples - len(trim_sound)
            offset = padding // 2
            trim_sound = np.pad(trim_sound, (offset, padding - offset), "constant")
        else:
            trim_sound = trim_sound[:min_samples]

        return trim_sound, sr
    
class MelSpectrogram(SerializableTransformMixin, AudioTransform):
    """Shifting time axis"""
    def __init__(self, parameters, always_apply=False, p=0.5):
        super(MelSpectrogram, self).__init__(always_apply, p)

        self.parameters = parameters
    
    def apply(self, data, **params):
        sound, sr = data

        melspec = librosa.feature.mfcc(y=sound, sr=sr, **self.parameters)
        melspec = librosa.power_to_db(melspec)
        melspec = melspec.astype(np.float32)

        return melspec, sr
    
class SpecAugment(SerializableTransformMixin, AudioTransform):
    """Shifting time axis"""
    def __init__(self, num_mask=2, freq_masking=0.15, time_masking=0.20, always_apply=False, p=0.5):
        super(SpecAugment, self).__init__(always_apply, p)

        self.num_mask = num_mask
        self.freq_masking = freq_masking
        self.time_masking = time_masking
    
    def apply(self, data, **params):
        melspec, sr = data

        spec_aug = self.spec_augment(melspec, 
                                     self.num_mask,
                                     self.freq_masking,
                                     self.time_masking,
                                     melspec.min())
        


        return spec_aug, sr
    
    def spec_augment(self, 
                    spec: np.ndarray,
                    num_mask=2,
                    freq_masking=0.15,
                    time_masking=0.20,
                    value=0):
        spec = spec.copy()
        num_mask = random.randint(1, num_mask)
        for i in range(num_mask):
            all_freqs_num, all_frames_num  = spec.shape
            freq_percentage = random.uniform(0.0, freq_masking)

            num_freqs_to_mask = int(freq_percentage * all_freqs_num)
            f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
            f0 = int(f0)
            spec[f0:f0 + num_freqs_to_mask, :] = value

            time_percentage = random.uniform(0.0, time_masking)

            num_frames_to_mask = int(time_percentage * all_frames_num)
            t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
            t0 = int(t0)
            spec[:, t0:t0 + num_frames_to_mask] = value

        return spec

class SpectToImage(SerializableTransformMixin, AudioTransform):

    def __init__(self, always_apply=False, p=0.5):
        super(SpectToImage, self).__init__(always_apply, p)
    
    def apply(self, data, **params):
        image, sr = data
        delta = librosa.feature.delta(image)
        accelerate = librosa.feature.delta(image, order=2)
        image = np.stack([image, delta, accelerate], axis=-1)
        image = image.astype(np.float32) / 100.0

        return image