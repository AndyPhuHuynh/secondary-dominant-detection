import librosa
import numpy as np
from pathlib import Path

import src.constants as c
from src.features.extractor import FeatureExtractor
from src.features.utils import load_audio_file, get_chord_segments

NUM_MFCCS = 13
NUM_MFCC_STATS = 2


def _validate_stat_index(stat_index):
    if stat_index < 0 or stat_index > NUM_MFCC_STATS:
        raise ValueError(f"Invalid sta index {stat_index}. Index must be in the range [0, {NUM_MFCC_STATS - 1}]")


def mfcc_stat_index_to_str(stat_index):
    _validate_stat_index(stat_index)
    names = ["mean", "std"]
    return names[stat_index]


def mfcc_feature_index(mfcc_index, chord_index, stat_index):
    if mfcc_index < 0 or mfcc_index >= NUM_MFCCS:
        raise ValueError(f"Invalid mfcc index {mfcc_index}. Index must be in the range [0, {NUM_MFCCS - 1}]")
    if chord_index < 0 or chord_index > c.NUM_CHORDS:
        raise ValueError(f"Invalid chord index {chord_index}. Index must be in the range [0, {c.NUM_CHORDS - 1}]")
    _validate_stat_index(stat_index)
    return (
        mfcc_index * c.NUM_CHORDS * NUM_MFCC_STATS +
        chord_index * NUM_MFCC_STATS +
        stat_index
    )


def extract_mfcc_features(signal):
    mfcc = librosa.feature.mfcc(y=signal, sr=c.SAMPLE_RATE, n_mfcc=NUM_MFCCS)

    mean = np.mean(mfcc, axis=1)
    std = np.std(mfcc, axis=1)

    return np.concatenate((mean, std))


class GlobalMFCCExtractor(FeatureExtractor):
    FEATURE_NAME = "global-mfcc"


    @classmethod
    def extract_features_from_file(cls, filepath: Path):
        signal = load_audio_file(filepath)
        return extract_mfcc_features(signal)


class PerChordMFCCExtractor(FeatureExtractor):
    FEATURE_NAME = "per-chord-mfcc"


    @classmethod
    def extract_features_from_file(cls, filepath: Path):
        signal = load_audio_file(filepath)
        segments = get_chord_segments(signal)
        features = np.array([extract_mfcc_features(seg) for seg in segments])
        return features.flatten()