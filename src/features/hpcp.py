import numpy as np
import librosa
import warnings
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

import src.constants as c
import src.paths as paths
from src.features.labels import get_label_string_to_num
from src.features.utils import load_audio_file, get_chord_segments


def compute_hpcp(signal):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        chroma = librosa.feature.chroma_cqt(y=signal, sr=c.SAMPLE_RATE)
    hpcp_mean = np.mean(chroma, axis=1)
    hpcp_std  = np.std(chroma, axis=1)
    return hpcp_mean, hpcp_std


def extract_hpcp_from_dataset(sample_rate: int):
    X = []
    y = []

    for path in paths.DATA_DIR.iterdir():
        if not path.is_dir():
            continue

        label_num = get_label_string_to_num(path.name)
        for filepath in tqdm(list(path.iterdir()), desc=f"Extracting hpcp for {path.name:<15}"):
            try:
                signal = load_audio_file(filepath)
                segments = get_chord_segments(signal)

                hpcp_means = []
                hpcp_stds = []
                for seg in segments:
                    hpcp_mean, hpcp_std = compute_hpcp(seg)
                    hpcp_means.append(hpcp_mean)
                    hpcp_stds.append(hpcp_std)
                hpcp_means = np.array(hpcp_means)
                hpcp_stds = np.array(hpcp_stds)

                transitions = []
                for i in range(7):
                    chord1 = hpcp_means[i]
                    chord2 = hpcp_means[i+1]
                    distance = np.linalg.norm(chord1 - chord2)
                    transitions.append(distance)
                transitions = np.array(transitions)

                similarities = []
                for i in range(7):
                    sim = cosine_similarity(
                        hpcp_means[i].reshape(1, -1),
                        hpcp_means[i + 1].reshape(1, -1)
                    )[0][0]
                    similarities.append(sim)
                similarities = np.array(similarities)

                roots = []
                for i in range(7):
                    root1 = np.argmax(hpcp_means[i])
                    root2 = np.argmax(hpcp_means[i+1])
                    interval = (root2 - root1) % 12
                    roots.append(interval)
                roots = np.array(roots)

                feature_vec = np.concatenate((
                    hpcp_means.flatten(),
                    hpcp_stds.flatten(),
                    similarities,
                    transitions,
                    roots
                ))

                X.append(feature_vec)
                y.append(label_num)
            except Exception as e:
                print(f"Unable to process file '{filepath}': {e}")
    X = np.array(X)
    y = np.array(y)

    return X, y