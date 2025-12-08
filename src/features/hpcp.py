import numpy as np
import librosa
from tqdm import tqdm

import src.paths as paths
from src.features.labels import get_label_string_to_num


def get_chord_segments(signal, sr):
    chord_length = 2
    num_chords = 8
    samples_per_chord = chord_length * sr
    segments = []
    for i in range(num_chords):
        start = i * samples_per_chord
        end = start + samples_per_chord
        segments.append(signal[start:end])

    return segments


def compute_hpcp(signal, sr):
    chroma = librosa.feature.chroma_cqt(y=signal, sr=sr)
    hpcp_mean = np.mean(chroma, axis=1)
    hpcp_std  = np.std(chroma, axis=1)
    return np.concatenate([hpcp_mean, hpcp_std])


def extract_hpcp_from_dataset(sample_rate: int):
    X = []
    y = []

    for path in paths.DATA_DIR.iterdir():
        if not path.is_dir():
            continue

        label_num = get_label_string_to_num(path.name)
        for filepath in tqdm(list(path.iterdir()), desc=f"Extracting hpcp for {path.name:<15}"):
            try:
                EXPECTED_SAMPLES = 16 * sample_rate # Eights chords each lasting for two seconds
                signal, sr = librosa.load(filepath, sr=sample_rate)
                signal = signal[:EXPECTED_SAMPLES]
                segments = get_chord_segments(signal, sr)

                hpcp_vectors = np.array([compute_hpcp(seg,sr) for seg in segments])
                # delta_hpcp = hpcp_vectors[1:] - hpcp_vectors[:-1]

                feature_vec = hpcp_vectors.flatten()
                # delta_hpcp = delta_hpcp.flatten()

                X.append(feature_vec)
                y.append(label_num)
            except Exception as e:
                print(f"Unable to process file '{filepath}': {e}")
    X = np.array(X)
    y = np.array(y)

    return X, y