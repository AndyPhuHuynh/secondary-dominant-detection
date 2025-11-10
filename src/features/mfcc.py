import os
import random
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.dataset import Dataset
from src.paths import SAMPLES_DIR, DATA_DIR
from src.features.labels import get_label_from_effect


def extract_mfcc(file_path: str, sample_rate: int, n_mfcc: int = 13):
    try:
        signal, sr = librosa.load(file_path, sr=sample_rate)
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        return mfcc_mean
    except Exception as e:
        print(f"Unable to process file '{file_path}': {e}")


def extract_mfcc_from_dataset(sample_rate: int, n_mfcc: int = 13) -> tuple[StandardScaler, Dataset]:
    """
    Extracts MFCC features and effect labels from the dataset
    :return: (StandardScaler, Dataset) A dataset containing MFCC features and genre labels along with its scaler
    """
    X = []
    y = []
    effects = os.listdir(DATA_DIR)
    for effect in effects:
        effect_dir = os.path.join(DATA_DIR, effect)
        if not os.path.isdir(effect_dir):
            continue

        print(f"Extracting MFCC for {effect}")
        for filename in os.listdir(effect_dir):
            file_path = os.path.join(effect_dir, filename)
            try:
                signal, sr = librosa.load(file_path, sr=sample_rate)
                mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
                mfcc_mean = np.mean(mfcc.T, axis=0)

                X.append(mfcc_mean)
                y.append(get_label_from_effect(effect))
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