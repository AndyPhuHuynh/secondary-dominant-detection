import librosa
import librosa.feature
import numpy as np
from tqdm import tqdm

import src.paths as paths
from src.features.labels import get_label_string_to_num

_NUM_CHORDS = 8


def split_feature_into_chord_alignment(unsplit_feature, frames_per_chord):
    features = []
    for chord_idx in range(_NUM_CHORDS):
        start = chord_idx * frames_per_chord
        end = start + frames_per_chord

        if start >= unsplit_feature.shape[1]:
            features.extend([0]*unsplit_feature.shape[0])
            continue

        chord_frames = unsplit_feature[:, start:end]

        if chord_frames.shape[1] == 0:
            features.extend([0]*unsplit_feature.shape[0])
        else:
            chord_mean = chord_frames.mean(axis=1)
            features.extend(chord_mean)
    return features


def extract_chord_aligned_features(signal, sr):
    hop_length = 512
    seconds_per_chord = 2.0

    frame_duration = hop_length / sr
    frames_per_chord = int(seconds_per_chord / frame_duration)

    signal, _ = librosa.effects.trim(signal, top_db=30)
    chroma = librosa.feature.chroma_cqt(y=signal, sr=sr, hop_length=hop_length)

    features = []
    split_chroma = split_feature_into_chord_alignment(chroma, frames_per_chord)

    tonnetz = librosa.feature.tonnetz(chroma=chroma)
    tonnetz_mean = np.mean(tonnetz, axis=1)
    tonnetz_std = np.std(tonnetz, axis=1)

    features.extend(split_chroma)

    return np.concatenate((split_chroma, tonnetz_mean, tonnetz_std))


def extract_stft_from_dataset(sample_rate: int):
    X = []
    y = []
    for path in paths.DATA_DIR.iterdir():
        if not path.is_dir():
            continue
        label = get_label_string_to_num(path.name)

        for filepath in tqdm(list(path.iterdir()), desc=f"Extracting hpcp for {label:<5}"):
            try:
                signal, sr = librosa.load(filepath, sr=sample_rate)
                feature_vec = extract_chord_aligned_features(signal, sr)
                X.append(feature_vec)
                y.append(label)
            except Exception as e:
                print(f"Unable to process file '{filepath}': {e}")
    X = np.array(X)
    y = np.array(y)
    return X, y
