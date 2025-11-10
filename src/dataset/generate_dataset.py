import os
import numpy as np
import soundfile as sf
import random
from tqdm import tqdm
from pedalboard import Pedalboard, Reverb, Distortion, Chorus

import src.frequencies
from src.paths import SAMPLES_DIR
from src.datatypes import WaveformType

SAMPLE_RATE = 44100
DURATION = 2.0 # seconds
os.makedirs(SAMPLES_DIR, exist_ok=True)


def sine_wave_transform(t, freq):
    return np.sin(2 * np.pi * freq * t)


def square_wave_transform(t, freq):
    return np.sign(sine_wave_transform(t, freq))


def triangle_wave_transform(t, freq):
    phase = t * freq
    return 2 * np.abs(2 * (phase - np.floor(phase + 0.5))) - 1


def sawtooth_wave_transform(t, freq):
    phase = t * freq
    return 2 * (phase - np.floor(0.5 + phase))


def waveform_transform(waveform: WaveformType, t, freq):
    match waveform:
        case WaveformType.Sine: return sine_wave_transform(t, freq)
        case WaveformType.Square: return square_wave_transform(t, freq)
        case WaveformType.Triangle: return triangle_wave_transform(t, freq)
        case WaveformType.Sawtooth: return sawtooth_wave_transform(t, freq)


def generate_tone(waveform: WaveformType):
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    frequencies = np.random.choice(
        [src.frequencies.A3,
            src.frequencies.A4,
            src.frequencies.E5,
            src.frequencies.A5,
            src.frequencies.E6
        ], size=np.random.randint(2, 4), replace=False)
    wave = np.zeros_like(t)
    for f in frequencies:
        wave += waveform_transform(waveform, t, f)
    wave /= len(frequencies)
    return wave.astype(np.float32)


def generate_dataset(num_samples: int = 50):
    for i in tqdm(range(num_samples), desc="Generating dataset"):
        for waveform in WaveformType:
            dry = generate_tone(waveform)

            reverb_board = Pedalboard([Reverb(room_size=1)])
            reverb = reverb_board(dry, SAMPLE_RATE)

            distortion_board = Pedalboard([Distortion(drive_db=random.uniform(10, 25))])
            distortion = distortion_board(dry, SAMPLE_RATE)

            dry_path = os.path.join(SAMPLES_DIR, f'dry/dry_{waveform.name.lower()}_{i:03d}.wav')
            reverb_path = os.path.join(SAMPLES_DIR, f'reverb/reverb_{waveform.name.lower()}_{i:03d}.wav')
            distortion_path = os.path.join(SAMPLES_DIR, f'distortion/distortion_{waveform.name.lower()}_{i:03d}.wav')

            os.makedirs(os.path.dirname(dry_path), exist_ok=True)
            os.makedirs(os.path.dirname(reverb_path), exist_ok=True)
            os.makedirs(os.path.dirname(distortion_path), exist_ok=True)

            sf.write(dry_path, dry, SAMPLE_RATE)
            sf.write(reverb_path, reverb, SAMPLE_RATE)
            sf.write(distortion_path, distortion, SAMPLE_RATE)