import os
import random
import librosa
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.paths import DATA_DIR
from src.features.labels import get_label_from_effect

def extract_mfcc(file_path: str, sample_rate: int, n_mfcc: int = 13):
    try:
        signal, sr = librosa.load(file_path, sr=sample_rate)
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        return mfcc_mean
    except Exception as e:
        print(f"Unable to process file '{file_path}': {e}")


def split_dataset(X: np.ndarray, y: np.ndarray) -> \
        tuple[np.ndarray, np.ndarray, \
                np.ndarray, np.ndarray, \
                np.ndarray, np.ndarray]:
    """
    Splits dataset into training, validation, and test sets
    :param X: Features of the dataset
    :param y: Labels of the dataset
    :return: (X_train, y_train, X_val, y_val, X_test, y_test): training, validation, and test sets
    """
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")

    data_len = len(X)
    shuffled_indices = [i for i in range(data_len)]
    random.shuffle(shuffled_indices)

    training_end_index: int = int(data_len * 0.6)
    validation_end_index: int = training_end_index + int(data_len * 0.2)

    train_indices      = shuffled_indices[:training_end_index]
    validation_indices = shuffled_indices[training_end_index:validation_end_index]
    test_indices       = shuffled_indices[validation_end_index:]

    X_train, y_train = X[train_indices],      y[train_indices]
    X_val,   y_val   = X[validation_indices], y[validation_indices]
    X_test,  y_test  = X[test_indices],       y[test_indices]

    return X_train, y_train, X_val, y_val, X_test, y_test


def extract_mfcc_from_dataset(sample_rate: int, n_mfcc: int = 13) -> tuple[StandardScaler, np.ndarray, np.ndarray]:
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
                y.append(get_label_from_effect(label))
            except Exception as e:
                print(f"Unable to process file '{filepath}': {e}")
    X = np.array(X)
    y = np.array(y)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return scaler, X, y