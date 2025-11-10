from enum import IntEnum


class WaveformType(IntEnum):
    Sine     = 0,
    Square   = 1,
    Triangle = 2,
    Sawtooth = 3,


from enum import IntEnum

class MidiNote(IntEnum):
    C0       = 12,
    C_sharp0 = 13,
    D_flat0  = 13
    D0       = 14,
    D_sharp0 = 15,
    E_flat0  = 15
    E0       = 16,
    E_sharp0 = 17,
    F_flat0  = 16
    F0       = 17,
    F_sharp0 = 18,
    G_flat0  = 18,
    G0       = 19,
    G_sharp0 = 20,
    A_flat0  = 20,
    A0       = 21,
    A_sharp0 = 22,
    B_flat0  = 22,
    B0       = 23,
    B_sharp0 = 24,

    C_flat4  = 59
    C4       = 60,
    C_sharp4 = 61,
    D_flat4  = 61,
    D4       = 62,
    D_sharp4 = 63,
    E_flat4  = 63,
    E4       = 64,
    E_sharp4 = 65,
    F_flat   = 64,
    F4       = 65,
    F_sharp4 = 66,
    G_flat4  = 66,
    G4       = 67,
    G_sharp4 = 68,
    A_flat4  = 68,
    A4       = 69,
    A_sharp4 = 70,
    B_flat4  = 70,
    B4       = 71,
    B_sharp4 = 72,


def midi_to_frequency(note: MidiNote) -> float:
    return 440.0 * (2 ** ((note - 69) / 12))