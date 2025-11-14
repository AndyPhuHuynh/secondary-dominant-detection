import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"

import subprocess
from tqdm import tqdm

import src.paths as paths
from src.setup.soundfonts import download_soundfonts

from src.features.labels import get_effect_from_label
from src.features.mfcc import extract_mfcc_from_dataset
from src.models.model1 import train_model1

from src.music_generation.generate_music import *

NUM_SONGS: int = 50

def generate_songs():
    os.makedirs(paths.DIATONIC_DIR, exist_ok=True)
    os.makedirs(paths.NON_DIATONIC_DIR, exist_ok=True)
    for i in tqdm(range(NUM_SONGS), desc="Generating songs"):
        diatonic_midi = paths.DIATONIC_DIR/f"diatonic_{i:03}.mid"
        diatonic_wave = paths.DIATONIC_DIR/f"diatonic_{i:03}.wav"
        non_diatonic_midi = paths.NON_DIATONIC_DIR/f"diatonic_{i:03}.mid"
        non_diatonic_wave = paths.NON_DIATONIC_DIR/f"diatonic_{i:03}.wav"

        diatonic_song = Song(is_diatonic=True)
        diatonic_song.write(diatonic_midi)
        subprocess.run([
            "fluidsynth", "-ni",
            "-F", diatonic_wave,
            "-r", "44100",
            paths.FLUID_SF_PATH,
            diatonic_midi
        ], check=True, stdout=subprocess.DEVNULL)
        diatonic_midi.unlink()

        print(f"Progression for {diatonic_midi}:")
        for phrase in diatonic_song.phrases:
            print(f"\t{phrase.progression}")

        non_diatonic_song = Song(is_diatonic=False)
        non_diatonic_song.write(non_diatonic_midi)
        subprocess.run([
            "fluidsynth", "-ni",
            "-F", non_diatonic_wave,
            "-r", "44100",
            paths.FLUID_SF_PATH,
            non_diatonic_midi
        ], check=True, stdout=subprocess.DEVNULL)
        non_diatonic_midi.unlink()

        print(f"Progression for {non_diatonic_midi}:")
        for phrase in non_diatonic_song.phrases:
            print(f"\t{phrase.progression}")




download_soundfonts()
# generate_songs()
scaler, dataset = extract_mfcc_from_dataset(44100)

model, history, ratios = train_model1(dataset)
for effect, accuracy in ratios.items():
    print(f"Accuracy for {get_effect_from_label(effect):<10} is {100 * accuracy:.2f}%")
