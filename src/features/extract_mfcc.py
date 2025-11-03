import os
import random
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.dataset import Dataset
from src.paths import GTZAN_DATA_DIR, GTZAN_CACHE_DIR
from src.features.labels import get_label

SAMPLE_RATE = 22050
N_MFCC = 13


def extract_single_feature(file_path: str, scaler: StandardScaler):
    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    mfcc_scaled = scaler.transform(mfcc_mean.reshape(1, -1))
    return mfcc_scaled


def extract_gtzan_features() -> tuple[StandardScaler, Dataset]:
    """
    Extracts MFCC features and genre labels from the GTZAN dataset
    :return: (StandardScaler, Dataset) A dataset containing MFCC features and genre labels along with its scaler
    """
    os.makedirs(GTZAN_CACHE_DIR, exist_ok=True)
    X = []
    y = []
    genres = os.listdir(GTZAN_DATA_DIR)
    for genre in genres:
        genre_dir = os.path.join(GTZAN_DATA_DIR, genre)
        cache_dir = os.path.join(GTZAN_CACHE_DIR, genre)
        os.makedirs(cache_dir, exist_ok=True)
        if not os.path.isdir(genre_dir):
            continue

        print(f"Extracting MFCC for {genre}")
        for filename in os.listdir(genre_dir):
            cache_path = os.path.join(cache_dir, filename) + ".npz"
            if os.path.exists(cache_path):
                data = np.load(cache_path)
                X.append(data["mfcc_mean"])
                y.append(get_label(genre))
                continue

            file_path = os.path.join(genre_dir, filename)
            try:
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC)
                mfcc_mean = np.mean(mfcc.T, axis=0)

                np.savez(cache_path, mfcc_mean=mfcc_mean)
                X.append(mfcc_mean)
                y.append(get_label(genre))
            except Exception as e:
                print(f"Unable to process file '{file_path}': {e}")
    X = np.array(X)
    y = np.array(y)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return scaler, Dataset(X, y)


def split_dataset(dataset: Dataset) -> tuple[Dataset, Dataset, Dataset]:
    """
    Splits dataset into training, validation, and test sets
    :param dataset: The dataset to split
    :return: (Dataset, Dataset, Dataset): training, validation, and test sets
    """
    data_len = len(dataset)
    shuffled_indices = [i for i in range(data_len)]
    random.shuffle(shuffled_indices)

    training_end_index: int = int(data_len * 0.6)
    validation_end_index: int = training_end_index + int(data_len * 0.2)

    train_indices      = shuffled_indices[:training_end_index]
    validation_indices = shuffled_indices[training_end_index:validation_end_index]
    test_indices       = shuffled_indices[validation_end_index:]

    X_train, y_train = dataset.X[train_indices],      dataset.y[train_indices]
    X_val,   y_val   = dataset.X[validation_indices], dataset.y[validation_indices]
    X_test,  y_test  = dataset.X[test_indices],       dataset.y[test_indices]

    return Dataset(X_train, y_train), Dataset(X_val, y_val), Dataset(X_test, y_test)



