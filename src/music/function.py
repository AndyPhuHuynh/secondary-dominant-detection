import random
from enum import StrEnum

class Function(StrEnum):
    Tonic             = "Tonic",
    TonicLike         = "TonicLike",
    Subdominant       = "Subdominant",
    Dominant          = "Dominant"
    SecondaryDominant = "SecondaryDominant"


    def rand_roman_numeral(self) -> str:
        if self == Function.Tonic:
            return "I"
        if self == Function.TonicLike:
            return random.choice(["iii", "vi"])
        if self == Function.Subdominant:
            return random.choice(["ii", "IV"])
        if self == Function.Dominant:
            return random.choice(["V", "viio"])
        raise ValueError(f"Unable to get roman numeral for {self}")


type PredecessorMap = dict[Function, list[Function]]

diatonic_predecessors: PredecessorMap = {
    Function.Tonic:       [Function.Subdominant, Function.Dominant],
    Function.TonicLike:   [Function.Tonic, Function.Subdominant, Function.Dominant],
    Function.Subdominant: [Function.Tonic, Function.TonicLike],
    Function.Dominant:    [Function.Tonic, Function.TonicLike, Function.Subdominant],
}

non_diatonic_predecessors: PredecessorMap = {
    Function.Tonic:       [Function.Subdominant, Function.Dominant],
    Function.TonicLike:   [Function.Tonic, Function.Subdominant, Function.Dominant, Function.SecondaryDominant],
    Function.Subdominant: [Function.Tonic, Function.TonicLike, Function.SecondaryDominant],
    Function.Dominant:    [Function.Tonic, Function.TonicLike, Function.Subdominant, Function.SecondaryDominant],
    Function.SecondaryDominant: [Function.Tonic, Function.TonicLike, Function.Subdominant, Function.Dominant, Function.SecondaryDominant],
}


def generate_progression(
    predecessors: PredecessorMap,
    length: int,
    target_chord: Function,
) -> list[Function]:
    if length < 4:
        raise ValueError("Length of chord progression must be at least 4")
    progression = [target_chord]
    next_chord = target_chord
    for i in range(1, length):
        prev_chord = random.choice(predecessors[next_chord])
        progression.insert(0, prev_chord)
        next_chord = prev_chord
    return progression


def generate_diatonic_progression(
    length: int,
    target_chord: Function,
) -> list[Function]:
    return generate_progression(diatonic_predecessors, length, target_chord)


def generate_non_diatonic_progression(
    length: int,
    target_chord: Function,
) -> list[Function]:
    while True:
        progression = generate_progression(non_diatonic_predecessors, length, target_chord)
        # if we contain a secondary dominant return the progression
        if Function.SecondaryDominant in progression:
            return progression
        # otherwise, find a place to insert at least one secondary dominant
        # we cannot place a secondary dominant before the tonic chord (it will be already in the key)
        valid_targets = {Function.TonicLike, Function.Subdominant, Function.Dominant}
        candidate_indices = [
            i for i in range(length - 1)
            if progression[i + 1] in valid_targets
        ]

        if not candidate_indices:
            continue

        progression[random.choice(candidate_indices)] = Function.SecondaryDominant
        return progression
    raise RuntimeError("Unreachable code path")


def transform_into_roman_numerals(progression: list[Function]) -> list[str]:
    numerals: list[str] = []
    for i in range(len(progression) - 1, -1, -1):
        if progression[i] == Function.SecondaryDominant and not numerals:
            raise ValueError("SecondaryDominant found at the end of progression")
        if progression[i] == Function.SecondaryDominant:
            numerals.insert(0, f"V7/{numerals[0]}")
            continue
        numerals.insert(0, progression[i].rand_roman_numeral())
    return numerals

