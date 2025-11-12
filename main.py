import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"

from src.dataset.generate_dataset import SAMPLE_RATE
from src.features.labels import get_effect_from_label
from src.features.mfcc import extract_mfcc_from_dataset
from src.models.model1 import train_model1
from tqdm import tqdm

from src.music_generation.generate_music import *
from src.paths import HAPPY_DIR, SAD_DIR
from pathlib import Path
import subprocess

sf2_path = Path("./soundfonts/FluidR3_GM.sf2").resolve()

NUM_SONGS: int = 10

def generate_songs():
    os.makedirs(HAPPY_DIR, exist_ok=True)
    os.makedirs(SAD_DIR, exist_ok=True)
    for i in tqdm(range(NUM_SONGS), desc="Generating songs"):
        happy_mid = HAPPY_DIR/f"happy_{i:03}.mid"
        happy_wav = HAPPY_DIR/f"happy_{i:03}.wav"
        sad_mid = SAD_DIR/f"sad_{i:03}.mid"
        sad_wav = SAD_DIR/f"sad_{i:03}.wav"

        happy_song = Song()
        happy_song.write(happy_mid)
        subprocess.run([
            "fluidsynth",
            "-ni",
            "-F", happy_wav,
            "-r", "44100",
            sf2_path,
            happy_mid
        ], check=True, stdout=subprocess.DEVNULL)
        happy_mid.unlink()
        print(f"Progression for {happy_mid}:")
        for phrase in happy_song.phrases:
            print(f"\t{phrase.progression}")

        sad_song = Song()
        sad_song.write(sad_mid)
        subprocess.run([
            "fluidsynth",
            "-ni",
            "-F", sad_wav,
            "-r", "44100",
            sf2_path,
            sad_mid
        ], check=True, stdout=subprocess.DEVNULL)
        sad_mid.unlink()

generate_songs()
# scaler, dataset = extract_mfcc_from_dataset(SAMPLE_RATE)
#
# model, history, ratios = train_model1(dataset)
# for effect, accuracy in ratios.items():
#     print(f"Accuracy for {get_effect_from_label(effect):<10} is {100 * accuracy:.2f}%")