from src.paths import PROJECT_DIR

NOTES: dict[str, int] = {
    "Cb": -1, "C":  0,  "C#": 1,
    "Db": 1,  "D":  2,  "D#": 3,
    "Eb": 3,  "E":  4,  "E#": 5,
    "Fb": 4,  "F":  5,  "F#": 6,
    "Gb": 6,  "G":  7,  "G#": 8,
    "Ab": 8,  "A":  9,  "A#": 10,
    "Bb": 10, "B":  11, "B#": 12,
}

def generate_enum():
    for current_octave in range(-1, 10):
        for note, base_midi in NOTES.items():
            midi_value = base_midi + 12 * (current_octave + 1)
            if midi_value < 0: continue
            if midi_value > 127: break

            accidental: str = \
                "sharp" if note[-1] == "#" else \
                "flat"  if note[-1] == "b" else ""

            note_name = note[0]
            if accidental:
                note_name += f"_{accidental}"

            if current_octave == -1:
                note_name += f"_neg1"
            else:
                note_name += str(current_octave)

            print(f"note_name: {note_name:<12} {midi_value}")