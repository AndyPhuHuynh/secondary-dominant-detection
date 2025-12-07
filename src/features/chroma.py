import librosa
import librosa.feature
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import src.paths as paths
from src.features.labels import get_label_string_to_num


def extract_stft_from_dataset(sample_rate: int):
    X = []
    y = []
    for path in paths.DATA_DIR.iterdir():
        if not path.is_dir():
            continue
        label = get_label_string_to_num(path.name)

        for filepath in tqdm(list(path.iterdir()), desc=f"Extracting chroma for {label:<5}"):
            try:
                signal, sr = librosa.load(filepath, sr=sample_rate)
                stft = librosa.feature.chroma_cqt(y=signal, sr=sr)

                stft_mean = np.mean(stft, axis=1)
                stft_std  = np.std(stft, axis=1)
                stft_min_ = np.min(stft, axis=1)
                stft_max_ = np.max(stft, axis=1)

                tonnetz = librosa.feature.tonnetz(y=signal, sr=sr)
                tonnetz_mean = np.mean(tonnetz, axis=1)
                tonnetz_std = np.std(tonnetz, axis=1)

                contrast = librosa.feature.spectral_contrast(y=signal, sr=sr)
                contrast_mean = np.mean(contrast, axis=1)
                contrast_std = np.std(contrast, axis=1)

                feature_vec = np.concatenate((
                    stft_mean, stft_std, stft_min_, stft_max_,
                    tonnetz_mean, tonnetz_std,
                    contrast_mean, contrast_std,
                ))

                X.append(feature_vec)
                y.append(label)
            except Exception as e:
                print(f"Unable to process file '{filepath}': {e}")
    X = np.array(X)
    y = np.array(y)
    return X, y
