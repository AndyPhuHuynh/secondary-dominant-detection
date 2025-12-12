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