from email._header_value_parser import Phrase

import music21
import random
from midiutil import MIDIFile
from pathlib import Path

from src.music.function import (Function,
                                generate_diatonic_progression,
                                generate_non_diatonic_progression,
                                transform_into_roman_numerals)

"""
1. Choose a random key
2. Choose a random chord progression
3. Generate a melody with the chord progression
"""

NUM_TRACKS:   int = 3
CHORD_TRACK:  int = 0
BASS_TRACK:   int = 1
MELODY_TRACK: int = 2

KEYS: list[music21.key.Key] = [music21.key.Key(k) for k in ["C", "E", "F", "G"]]

RHYTHM_PATTERNS = [
    [1, 1, 1, 1],
    [1.5, 1.5, 1],
    [0.5, 0.5, 1, 2],
    [1.5, 0.5, 1, 1],
]

PHRASE_PATTERNS: list[list[str]] = [
    ["A", "B", "A", "B"],
    ["A", "B", "A", "C"],
    ["A", "A", "B", "A"]
]

def write_ending_chord(mf: MIDIFile, chord: music21.chord.Chord, channel: int, chord_start_time: float | int, volume: int):
    for pitch in chord.pitches:
        mf.addNote(CHORD_TRACK, channel, pitch.midi - 12, chord_start_time, 4, volume)
    mf.addNote(BASS_TRACK, channel, chord.root().midi - 24, chord_start_time, 4, volume)
    mf.addNote(MELODY_TRACK, channel, chord.root().midi, chord_start_time, 4, volume)
    return chord_start_time + 4


def generate_phases(is_diatonic: bool) -> list[Phrase]:
    pattern: list[str] = random.choice(PHRASE_PATTERNS)
    phrase_dict: dict[str, Phrase] = {}
    for symbol in pattern:
        if symbol in phrase_dict:
            continue
        phrase_dict[symbol] = Phrase(is_diatonic)
    return [phrase_dict[symbol] for symbol in pattern]


class Phrase:
    def __init__(self, is_diatonic: bool = True):
        if not is_diatonic:
            self.progression = transform_into_roman_numerals(generate_non_diatonic_progression(4, Function.Tonic))
        else:
            self.progression = transform_into_roman_numerals(generate_diatonic_progression(4, Function.Tonic))


class Song:
    key: music21.key.Key
    tonic_symbol: str
    phrases: list[Phrase]

    def __init__(self, is_diatonic: bool):
        self.key  = random.choice(KEYS)
        self.tonic_symbol = "I"
        self.phrases = generate_phases(is_diatonic)


    def write(self, filename: Path):
        channel = 0
        volume = 100
        bpm = 120
        mf = MIDIFile(NUM_TRACKS)
        for track in range(NUM_TRACKS):
            mf.addTempo(track, channel, bpm)

        time = 0
        for phrase in self.phrases:
            chord_start_time = time
            for roman_numeral in phrase.progression:
                chord = music21.roman.RomanNumeral(roman_numeral, self.key)

                for pitch in chord.pitches:
                    mf.addNote(0, channel, pitch.midi - 12, chord_start_time, 4, volume)
                mf.addNote(1, channel, chord.root().midi - 24, chord_start_time, 4, volume)
                chord_start_time += 4
                time = generate_melody_for_chord(chord, time, 2, channel, volume, mf)

        if self.phrases[-1].progression[-1] != self.tonic_symbol and random.choice([True, False]):
            write_ending_chord(mf, music21.roman.RomanNumeral(self.tonic_symbol, self.key), channel, time, volume)

        with open(filename, "wb") as f:
            mf.writeFile(f)


def generate_melody_for_chord(chord, start_time, track, channel, volume, midi_file):
    rhythm = random.choice(RHYTHM_PATTERNS)
    beat = 0
    for duration in rhythm:
        note = random.choice(chord.pitches).midi
        midi_file.addNote(track, channel, note, start_time + beat, duration, volume)
        beat += duration
    return start_time + beat

