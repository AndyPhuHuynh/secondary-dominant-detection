import music21
import random
from midiutil import MIDIFile
from pathlib import Path

import src.paths as paths
from src.music.generation import (
    generate_diatonic_progression,
    generate_non_diatonic_progression,
    generate_roman_numerals,
    midi_to_wave
)

KEYS: list[music21.key.Key] = [music21.key.Key(k) for k in ["C", "E", "F", "G"]]

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

        midi_to_wave(midi_path, wave_path, paths.FLUID_SF_PATH)
        midi_path.unlink()


    def string_info(self) -> str:
        info = f"{str(self.key):>2}"
        for numeral in self.progression:
            info += f" {numeral:>5}"
        return info

