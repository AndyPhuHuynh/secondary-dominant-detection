import music21
import random
from midiutil import MIDIFile

"""
1. Choose mood (happy or sad)
2. Pick a chord progression from a set list for each mood
3. Generate a melody with the chord progression
"""

maj_progression = ["I", "IV", "V", "vi"]
min_progression = ["i", "iv", "v", "VI"]

RHYTHM_PATTERNS = [
    [1, 1, 1, 1],
    [1.5, 1.5, 1],
    [0.5, 0.5, 1, 2],
    [1.5, 0.5, 1, 1],
]

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


def generate_melody(key: music21.key.Key, progression, filename):
    channel = 0
    volume = 100
    bpm = 120
    mf = MIDIFile(3) # 0: chords, 1: bass, 2: melody
    for track in range(3):
        mf.addTempo(track, channel, bpm)

    time = 0
    for roman_numeral in progression:
        chord_start_time = time
        chord = music21.roman.RomanNumeral(roman_numeral, key)

        for pitch in chord.pitches:
            mf.addNote(0, channel, pitch.midi - 12, chord_start_time, 4, volume)
        mf.addNote(1, channel, chord.root().midi - 24, chord_start_time, 4, volume)
        time = generate_melody_for_chord(chord, key.getScale(), time, 2, channel, volume, mf)

    with open(filename, "wb") as f:
        mf.writeFile(f)


