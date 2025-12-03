import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"

import shutil
from tqdm import tqdm

import src.paths as paths
from src.setup.soundfonts import download_soundfonts

from src.features.labels import get_effect_from_label
from src.features.mfcc import extract_mfcc_from_dataset
from src.models.model1 import train_model1

from src.music.song import Song

NUM_SONGS: int = 50

def generate_songs():
    paths.DIATONIC_INFO.unlink(missing_ok=True)
    paths.NON_DIATONIC_INFO.unlink(missing_ok=True)

    shutil.rmtree(paths.DIATONIC_DATA_DIR)
    shutil.rmtree(paths.NON_DIATONIC_DATA_DIR)
    paths.DIATONIC_DATA_DIR.mkdir(parents=True, exist_ok=True)
    paths.NON_DIATONIC_DATA_DIR.mkdir(parents=True, exist_ok=True)

    paths.INFO_DIR.mkdir(parents=True, exist_ok=True)
    with paths.DIATONIC_INFO.open("w") as diatonic_info, \
        paths.NON_DIATONIC_INFO.open("w") as non_diatonic_info:

        for i in tqdm(range(NUM_SONGS), desc="Generating songs"):
            diatonic_path     = paths.DIATONIC_DATA_DIR / f"diatonic_{i:03}.mid"
            non_diatonic_path = paths.NON_DIATONIC_DATA_DIR / f"non_diatonic_{i:03}.mid"

            diatonic_song = Song(is_diatonic=True)
            non_diatonic_song = Song(is_diatonic=False)

            diatonic_song.write(diatonic_path)
            non_diatonic_song.write(non_diatonic_path)

            diatonic_info.write(f"{diatonic_song.string_info()}\n")
            non_diatonic_info.write(f"{non_diatonic_song.string_info()}\n")


download_soundfonts()
generate_songs()
scaler, dataset = extract_mfcc_from_dataset(44100)
print(dataset.X.shape)

model, history, ratios = train_model1(dataset)
for effect, accuracy in ratios.items():
    print(f"Accuracy for {get_effect_from_label(effect):<10} is {100 * accuracy:.2f}%")