import librosa
from pathlib import Path
from sklearn.metrics import precision_score, recall_score

import src.constants as c


def load_audio_file(filepath: Path):
    signal, sr = librosa.load(filepath, sr=c.SAMPLE_RATE)
    signal = signal[:c.SAMPLES_PER_WAVE]
    return signal


def get_chord_segments(signal):
    if len(signal) < c.SAMPLES_PER_WAVE:
        raise ValueError(f"Signal too short: len of signal is {len(signal)}")
    segments = []
    for i in range(c.NUM_CHORDS):
        start = i * c.SAMPLES_PER_CHORD
        end = start + c.SAMPLES_PER_CHORD
        segments.append(signal[start:end])
    return segments


def evaluate_precision_and_recall(model, X, y):
    y_pred = model.predict(X)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    return precision, recall