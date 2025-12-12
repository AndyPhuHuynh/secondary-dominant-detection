import librosa.feature
import numpy as np
from pathlib import Path

import src.constants as c
from src.features.extractor import FeatureExtractor
from src.features.utils import load_audio_file, get_chord_segments


NUM_TONNETZ_AXIS = 3
NUM_TONNETZ_COEFFICIENTS = 6
NUM_TONNETZ_STATS = 2


def get_tonnetz_axis_name(axis: int):
    if axis < 0 or axis >= NUM_TONNETZ_AXIS:
        raise ValueError(f"Invalid tonnetz axis {axis}. Axis must be in the range [0, {NUM_TONNETZ_AXIS - 1}]")

    if axis == 0:
        return "Perfect Fifth Axis"
    elif axis == 1:
        return "Minor Third Axis"
    else:
        return "Major Third Axis"



def extract_tonnetz_features(signal):
   tonnetz = librosa.feature.tonnetz(y=signal, sr=c.SAMPLE_RATE)
   mean = np.mean(tonnetz, axis=1)
   std = np.std(tonnetz, axis=1)
   return np.concatenate((mean, std))


class GlobalTonnetzExtractor(FeatureExtractor):
    FEATURE_NAME = "global-tonnetz"


    @classmethod
    def extract_features_from_file(cls, filepath: Path):
        signal = load_audio_file(filepath)
        return extract_tonnetz_features(signal)


class PerChordTonnetzExtractor(FeatureExtractor):
    FEATURE_NAME = "per-chord-mfcc"


    @classmethod
    def extract_features_from_file(cls, filepath: Path):
        signal = load_audio_file(filepath)
        segments = get_chord_segments(signal)
        features = np.array([extract_tonnetz_features(seg) for seg in segments])
        return features.flatten()


class TonnetzContrastExtractor(FeatureExtractor):
    FEATURE_NAME = "tonnetz_contrast"

    @classmethod
    def extract_features_from_file(cls, filepath: Path):
        signal = load_audio_file(filepath)
        segments = get_chord_segments(signal)
        tonnetz = np.array([librosa.feature.tonnetz(y=seg, sr=c.SAMPLE_RATE).mean(axis=1) for seg in segments])

        delta = tonnetz[1:] - tonnetz[:-1]
        d = np.linalg.norm(delta, axis=1)

        mean_step = d.mean()
        max_step = d.max()
        std_step = d.std()
        sum_step = d.sum()
        step_stats = np.array([mean_step, max_step, std_step, sum_step])

        k = int(np.argmax(d))
        mu_pre = tonnetz[:k + 1].mean(axis=0)
        mu_post = tonnetz[k + 1:].mean(axis=0) if k + 1 < len(tonnetz) else tonnetz[-1]
        prepost = np.linalg.norm(mu_post - mu_pre)

        eps = 1e-8
        sorted_d = np.sort(d)
        peak_ratio = sorted_d[-1] / (d.mean() + eps)
        peak_minus_mean = sorted_d[-1] - d.mean()
        two_peak_sum = sorted_d[-1] + sorted_d[-2]
        resolution_contrast = np.array([peak_ratio, peak_minus_mean, two_peak_sum])

        return np.concatenate((step_stats, np.array([prepost]), resolution_contrast))

