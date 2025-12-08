import shutil
from tqdm import tqdm

import src.paths as paths
from src.music.song import Song


NUM_DEFAULT_SONGS = 200


def generate_songs(num_songs: int = 50):
    paths.INFO_DIATONIC_TXT.unlink(missing_ok=True)
    paths.INFO_NON_DIATONIC_TXT.unlink(missing_ok=True)

    if paths.DATA_DIATONIC_DIR.exists(): shutil.rmtree(paths.DATA_DIATONIC_DIR)
    if paths.DATA_NON_DIATONIC_DIR.exists(): shutil.rmtree(paths.DATA_NON_DIATONIC_DIR)
    paths.DATA_DIATONIC_DIR.mkdir(parents=True, exist_ok=True)
    paths.DATA_NON_DIATONIC_DIR.mkdir(parents=True, exist_ok=True)

    paths.INFO_DIR.mkdir(parents=True, exist_ok=True)
    with paths.INFO_DIATONIC_TXT.open("w") as diatonic_info, \
        paths.INFO_NON_DIATONIC_TXT.open("w") as non_diatonic_info:

        for i in tqdm(range(num_songs), desc="Generating songs"):
            diatonic_path     = paths.DATA_DIATONIC_DIR / f"diatonic_{i:03}.mid"
            non_diatonic_path = paths.DATA_NON_DIATONIC_DIR / f"non_diatonic_{i:03}.mid"

            diatonic_song = Song(is_diatonic=True)
            non_diatonic_song = Song(is_diatonic=False)

            diatonic_song.write(diatonic_path)
            non_diatonic_song.write(non_diatonic_path)

            diatonic_info.write(f"{diatonic_song.string_info()}\n")
            non_diatonic_info.write(f"{non_diatonic_song.string_info()}\n")


def setup_songs(num_songs: int = NUM_DEFAULT_SONGS, force_setup: bool = False):
    diatonic_songs     = list(paths.DATA_DIATONIC_DIR.glob("*.wav"))
    non_diatonic_songs = list(paths.DATA_NON_DIATONIC_DIR.glob("*.wav"))

    songs_exist = len(diatonic_songs) >= 1 and len(non_diatonic_songs) >= 1
    if not force_setup and songs_exist:
        return

    shutil.rmtree(paths.CACHE_DIR)
    generate_songs(num_songs)
