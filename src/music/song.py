import music21
import random
import shutil
from midiutil import MIDIFile
from tqdm import tqdm
from pathlib import Path

import src.paths as paths
from src.music.generation import (
    generate_diatonic_progression,
    generate_non_diatonic_progression,
    generate_roman_numerals,
)
from src.music.fluidsynth import midi_to_wave
from src.soundfonts import get_random_soundfont_preset


PITCHES = ['C', 'C#', 'D', 'E-', 'E', 'F', 'F#', 'G', 'A-', 'A', 'B-', 'B']
KEYS = [music21.key.Key(p, 'major') for p in PITCHES]

NUM_TRACKS:   int = 2
CHORD_TRACK:  int = 0
BASS_TRACK:   int = 1


class Song:
    key: music21.key.Key
    progression: list[str]

    def __init__(self, is_diatonic: bool):
        if is_diatonic:
            functions = generate_diatonic_progression(8)
        else:
            functions = generate_non_diatonic_progression(8)
        self.progression = generate_roman_numerals(functions)
        self.key = random.choice(KEYS)
        self.sf_preset = get_random_soundfont_preset()


    def write(self, path: Path) -> None:
        channel = 0
        volume = 100
        bpm = 120

        mf = MIDIFile(NUM_TRACKS)
        for track in range(NUM_TRACKS):
            mf.addTempo(track, channel, bpm)

        time = 0
        for numeral in self.progression:
            chord = music21.roman.RomanNumeral(numeral, self.key)

            mf.addNote(BASS_TRACK, channel, chord.root().midi - 24, time, 4, volume)
            for pitch in chord.pitches:
                mf.addNote(CHORD_TRACK, channel, pitch.midi - 12, time, 4, volume)
            time += 4

        midi_path = path.with_suffix(".mid")
        midi_path.parent.mkdir(parents=True, exist_ok=True)

        wave_path = path.with_suffix(".wav")
        wave_path.parent.mkdir(parents=True, exist_ok=True)

        with open(midi_path, "wb") as f:
            mf.writeFile(f)

        midi_to_wave(midi_path, wave_path,
            self.sf_preset["path"],
            bank=self.sf_preset["bank"],
            preset=self.sf_preset["preset"]
        )
        midi_path.unlink()


    def string_info(self) -> str:
        info = f"{self.sf_preset["name"]:>20} {self.sf_preset["preset"]:>4}  {str(self.key):>10}"
        for numeral in self.progression:
            info += f" {numeral:>10}"
        return info


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