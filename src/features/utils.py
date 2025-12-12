import librosa
from pathlib import Path

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
