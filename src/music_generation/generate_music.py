import music21
import random
from midiutil import MIDIFile
from pathlib import Path

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

type ChordTransitionMap = dict[str, dict[str, int]]
MAJOR_TRANSITIONS: ChordTransitionMap = {
    "I":    {"I":1, "ii":3, "iii":1, "IV":3, "V":2, "V7":2, "vi":2, "vii°":0},
    "ii":   {"I":1, "ii":1, "iii":0, "IV":2, "V":6, "V7":6, "vi":1, "vii°":3},
    "iii":  {"I":1, "ii":2, "iii":1, "IV":3, "V":2, "V7":2, "vi":5, "vii°":0},
    "IV":   {"I":3, "ii":2, "iii":0, "IV":2, "V":5, "V7":5, "vi":2, "vii°":0},
    "V":    {"I":8, "ii":0, "iii":0, "IV":1, "V":2, "V7":1, "vi":3, "vii°":0},
    "V7":   {"I":9, "ii":0, "iii":0, "IV":1, "V":1, "V7":1, "vi":3, "vii°":0},
    "vi":   {"I":2, "ii":3, "iii":1, "IV":4, "V":3, "V7":3, "vi":2, "vii°":1},
    "vii°": {"I":8, "ii":0, "iii":2, "IV":0, "V":1, "V7":1, "vi":0, "vii°":1},
}

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

def get_chord_progression_from_map(chord_map: ChordTransitionMap, start_chord: str, length: int):
    progression: list[str] = [start_chord]
    for i in range(length - 1):
        if not progression[i] in chord_map:
            raise ValueError(f"Chord not found in map: {progression[i]}")
        transitions = chord_map[progression[i]]
        chords, weights = zip(*transitions.items())
        progression.append(random.choices(chords, weights)[0])
    return progression


def write_ending_chord(mf: MIDIFile, chord: music21.chord.Chord, channel: int, chord_start_time: float | int, volume: int):
    for pitch in chord.pitches:
        mf.addNote(CHORD_TRACK, channel, pitch.midi - 12, chord_start_time, 4, volume)
    mf.addNote(BASS_TRACK, channel, chord.root().midi - 24, chord_start_time, 4, volume)
    mf.addNote(MELODY_TRACK, channel, chord.root().midi, chord_start_time, 4, volume)
    return chord_start_time + 4


def generate_phases():
    pattern: list[str] = random.choice(PHRASE_PATTERNS)
    phrase_dict: dict[str, Phrase] = {}
    for symbol in pattern:
        if symbol in phrase_dict:
            continue
        phrase_dict[symbol] = Phrase("I")
    return [phrase_dict[symbol] for symbol in pattern]


class Phrase:
    def __init__(self, start_chord: str):
        self.progression = get_chord_progression_from_map(MAJOR_TRANSITIONS, start_chord, 4)


class Song:
    key: music21.key.Key
    tonic_symbol: str
    phrases: list[Phrase]

    def __init__(self):
        self.key  = random.choice(KEYS)
        self.tonic_symbol = "I"
        self.phrases = generate_phases()


    def write(self, filename: Path):
        channel = 0
        volume = 100
        bpm = 120
        mf = MIDIFile(NUM_TRACKS)  # 0: chords, 1: bass, 2: melody
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
            #     time = generate_melody_for_chord(chord, self.key.getScale(), time, 2, channel, volume, mf)
            time = generate_melody_for_phrase(phrase, self.key.getScale(), time, MELODY_TRACK, channel ,volume, mf)

        if self.phrases[-1].progression[-1] != self.tonic_symbol and random.choice([True, False]):
            write_ending_chord(mf, music21.roman.RomanNumeral(self.tonic_symbol, self.key), channel, time, volume)

        with open(filename, "wb") as f:
            mf.writeFile(f)


def generate_melody_for_chord(chord, scale, start_time, track, channel, volume, midi_file):
    rhythm = random.choice(RHYTHM_PATTERNS)
    beat = 0
    for duration in rhythm:
        if beat in [0, 2]:
            note = random.choice(chord.pitches).midi
        else:
            note = random.choice(scale.pitches).midi
        midi_file.addNote(track, channel, note, start_time + beat, duration, volume)
        beat += duration
    return start_time + beat


def get_neighbor_in_scale(note: music21.note.Note, scale: music21.scale.ConcreteScale, direction: str) -> music21.note.Note:
    pitches = scale.getPitches(music21.pitch.Pitch("C0"), music21.pitch.Pitch("C8"))
    if direction == "up":
        for p in pitches:
            if p.midi > note.pitch.midi:
                return music21.note.Note(p)
    if direction == "down":
        for p in reversed(pitches):
            if p.midi < note.pitch.midi:
                return music21.note.Note(p)
    return note


def leap_from_note(note: music21.note.Note, scale: music21.scale.ConcreteScale, direction: str, leap_count: int):
    pitches = scale.getPitches(music21.pitch.Pitch("C0"), music21.pitch.Pitch("C8"))
    leaps = 0
    if direction == "up":
        for p in pitches:
            if p.midi > note.pitch.midi:
                leaps += 1
            if leaps == leap_count:
                return music21.note.Note(p)
    if direction == "down":
        for p in reversed(pitches):
            if p.midi < note.pitch.midi:
                leaps += 1
            if leaps == leap_count:
                return music21.note.Note(p)
    return note


def get_next_note(prev_note: music21.note.Note, current_chord: music21.chord.Chord, scale: music21.scale.ConcreteScale) -> music21.note.Note:
    probabilites = {"chord tone": 3, "neighbor": 2, "leap": 1}
    choices, weights = zip(*probabilites.items())
    choice = random.choices(choices, weights)[0]
    if choice == "chord tone":
        return music21.note.Note(random.choice(current_chord.pitches))
    if choice == "neighbor":
        neighbor = random.choice(["up", "down"])
        return get_neighbor_in_scale(prev_note, scale, neighbor)
    leap = random.choice(range(4, 6))
    return leap_from_note(prev_note, scale, leap, 4)


def generate_melody_for_phrase(phrase, scale, start_time, track, channel, volume, midi_file):
    rhythm = [1, 1, 1, 1]
    time = 0
    prev_note: music21.note.Note | None = None
    for roman_numeral in phrase.progression:
        chord = music21.roman.RomanNumeral(roman_numeral, scale)
        for duration in rhythm:
            if prev_note is None:
                prev_note = music21.note.Note(chord.pitches[0])
            else:
                prev_note = get_next_note(prev_note, chord, scale)
            midi_file.addNote(track, channel, prev_note.pitch.midi, start_time + time, duration, volume)
            time += duration
    return start_time + time

