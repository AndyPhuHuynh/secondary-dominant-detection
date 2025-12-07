import os
import random
import librosa
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.paths import DATA_DIR
from src.features.labels import get_label_string_to_num

def extract_mfcc(file_path: str, sample_rate: int, n_mfcc: int = 13):
    try:
        signal, sr = librosa.load(file_path, sr=sample_rate)
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        return mfcc_mean
    except Exception as e:
        print(f"Unable to process file '{file_path}': {e}")


def extract_mfcc_from_dataset(sample_rate: int, n_mfcc: int = 13) -> tuple[np.ndarray, np.ndarray]:
    """
    Extracts MFCC features and effect labels from the dataset
    :return: (StandardScaler, X, y) A dataset containing MFCC features and genre labels along with its scaler
    """
    X = []
    y = []
    labels = os.listdir(DATA_DIR)
    for label in labels:
        label_dir = os.path.join(DATA_DIR, label)
        if not os.path.isdir(label_dir):
            continue

        for filename in tqdm(os.listdir(label_dir), desc=f"Extracting MFCC for {label:<5}"):
            filepath = os.path.join(label_dir, filename)
            try:
                signal, sr = librosa.load(filepath, sr=sample_rate)
                mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)

                mean = np.mean(mfcc.T, axis=0)
                std  = np.std(mfcc.T, axis=0)
                min_ = np.min(mfcc.T, axis=0)
                max_ = np.max(mfcc.T, axis=0)

                feature_vec = np.concatenate((mean, std, min_, max_))
                X.append(feature_vec)
                y.append(get_label_string_to_num(label))
            except Exception as e:
                print(f"Unable to process file '{filepath}': {e}")
    X = np.array(X)
    y = np.array(y)

    return X, y