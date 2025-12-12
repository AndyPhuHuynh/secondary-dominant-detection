import librosa
import numpy as np
from pathlib import Path

import src.constants as c
from src.features.extractor import FeatureExtractor
from src.features.utils import load_audio_file


class GlobalMFCCExtractor(FeatureExtractor):
    FEATURE_NAME = "global-mfcc"


    @classmethod
    def extract_features_from_file(cls, filepath: Path):
        signal = load_audio_file(filepath)
        mfcc = librosa.feature.mfcc(y=signal, sr=c.SAMPLE_RATE, n_mfcc=c.N_MFCCS)

        mean = np.mean(mfcc.T, axis=0)
        std = np.std(mfcc.T, axis=0)
        min_ = np.min(mfcc.T, axis=0)
        max_ = np.max(mfcc.T, axis=0)

        return np.concatenate((mean, std, min_, max_))