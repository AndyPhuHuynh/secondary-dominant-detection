import numpy as np
import librosa.feature
import warnings
from pathlib import Path

import src.constants as c
from src.features.extractor import FeatureExtractor
from src.features.utils import load_audio_file, get_chord_segments


def extract_hpcp_features(signal):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        hpcp = librosa.feature.chroma_cqt(y=signal, sr=c.SAMPLE_RATE)
    hpcp_mean = np.mean(hpcp, axis=1)
    hpcp_std = np.std(hpcp, axis=1)
    return np.concatenate((hpcp_mean, hpcp_std))


def extract_hpcp_and_tonnetz_features(signal):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        hpcp = librosa.feature.chroma_cqt(y=signal, sr=c.SAMPLE_RATE)
        tonnetz = librosa.feature.tonnetz(y=signal, sr=c.SAMPLE_RATE)
    hpcp_mean = np.mean(hpcp, axis=1)
    tonnetz_mean = np.mean(tonnetz, axis=1)
    return np.concatenate((hpcp_mean, tonnetz_mean))


class HPCPExtractor(FeatureExtractor):
    FEATURE_NAME = "hpcp"

    @classmethod
    def extract_features_from_file(cls, filepath: Path):
        signal = load_audio_file(filepath)
        segments = get_chord_segments(signal)
        features = np.array([extract_hpcp_features(seg) for seg in segments])
        return features.flatten()


class HPCPAndTonnetzExtractor(FeatureExtractor):
    FEATURE_NAME = "hpcp-tonnetz"

    @classmethod
    def extract_features_from_file(cls, filepath: Path):
        signal = load_audio_file(filepath)
        segments = get_chord_segments(signal)
        features = np.array([extract_hpcp_and_tonnetz_features(seg) for seg in segments])
        return features.flatten()