import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"
import numpy as np

from src.dataset.generate_dataset import generate_dataset, SAMPLE_RATE
from src.features.labels import get_effect_from_label
from src.features.mfcc import extract_mfcc, extract_mfcc_from_dataset
from src.models.model1 import train_model1
from src.setup.generate_midi_enum import generate_enum
from src.visualization.mfcc_example import plot_examples_one_each
from src.datatypes import MidiNote, midi_to_frequency


# generate_dataset(100)
# scaler, dataset = extract_mfcc_from_dataset(SAMPLE_RATE)
#
# model, history, ratios = train_model1(dataset)
# for effect, accuracy in ratios.items():
#     print(f"Accuracy for {get_effect_from_label(effect):<10} is {100 * accuracy:.2f}%")
#
# plot_examples_one_each()
#
# while True:
#     file_path = input("Enter file path:")
#     if file_path == "quit":
#         break
#     try:
#         mfcc = extract_mfcc(file_path, SAMPLE_RATE)
#         mfcc = np.expand_dims(mfcc, axis=0)
#         prediction = model.predict(mfcc)
#         print(prediction)
#     except Exception as e:
#         print(e)

from src.music_generation.generate_music import *
from src.paths import HAPPY_DIR, SAD_DIR
from midi2audio import FluidSynth
from pathlib import Path
import subprocess

sf2_path = Path("./soundfonts/FluidR3_GM.sf2").resolve()
fs = FluidSynth(sf2_path)

NUM_SONGS: int = 10

def generate_songs():
    os.makedirs(HAPPY_DIR, exist_ok=True)
    os.makedirs(SAD_DIR, exist_ok=True)
    for i in range(NUM_SONGS):
        happy_mid = HAPPY_DIR/f"happy_{i:03}.mid"
        happy_wav = HAPPY_DIR/f"happy_{i:03}.wav"
        sad_mid = SAD_DIR/f"sad_{i:03}.mid"
        sad_wav = SAD_DIR/f"sad_{i:03}.wav"

        generate_melody(music21.key.Key("C"), maj_progression, happy_mid)
        generate_melody(music21.key.Key("c"), min_progression, sad_mid)

        subprocess.run([
            "fluidsynth",
            "-ni",
            "-F", happy_wav,
            "-r", "44100",
            "soundfonts/FluidR3_GM.sf2",
            happy_mid
        ], check=True, stdout=subprocess.DEVNULL)
        happy_mid.unlink()

        subprocess.run([
            "fluidsynth",
            "-ni",
            "-F", sad_wav,
            "-r", "44100",
            "soundfonts/FluidR3_GM.sf2",
            sad_mid
        ], check=True, stdout=subprocess.DEVNULL)
        sad_mid.unlink()

generate_songs()
scaler, dataset = extract_mfcc_from_dataset(SAMPLE_RATE)

model, history, ratios = train_model1(dataset)
for effect, accuracy in ratios.items():
    print(f"Accuracy for {get_effect_from_label(effect):<10} is {100 * accuracy:.2f}%")