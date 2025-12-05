import random

import src.music.function as fn

def generate_progression(
    predecessors: fn.PredecessorMap,
    length: int,
    target_chord: fn.Function | None = None,
) -> list[fn.Function]:
    if length < 4:
        raise ValueError("Length of chord progression must be at least 4")
    if target_chord is None:
        target_chord = fn.Function.Tonic
    progression = [target_chord]
    next_chord = target_chord
    for i in range(1, length):
        prev_chord = random.choice(predecessors[next_chord])
        progression.insert(0, prev_chord)
        next_chord = prev_chord
    return progression


def generate_diatonic_progression(
    length: int,
    target_chord: fn.Function | None = None,
) -> list[fn.Function]:
    return generate_progression(fn.diatonic_predecessors, length, target_chord)


def generate_non_diatonic_progression(
    length: int,
    target_chord: fn.Function | None = None,
) -> list[fn.Function]:
    while True:
        progression = generate_progression(fn.non_diatonic_predecessors, length, target_chord)
        # if we contain a secondary dominant return the progression
        if fn.Function.SecondaryDominant in progression:
            return progression
        # otherwise, find a place to insert at least one secondary dominant
        # we cannot place a secondary dominant before the tonic chord (it will be already in the key)
        valid_targets = {fn.Function.TonicLike, fn.Function.Subdominant, fn.Function.Dominant}
        candidate_indices = [
            i for i in range(length - 1)
            if progression[i + 1] in valid_targets
        ]

        if not candidate_indices:
            continue

        progression[random.choice(candidate_indices)] = fn.Function.SecondaryDominant
        return progression
    raise RuntimeError("Unreachable code path")


def generate_roman_numerals(progression: list[fn.Function]) -> list[str]:
    numerals: list[str] = []
    for i in range(len(progression) - 1, -1, -1):
        if progression[i] == fn.Function.SecondaryDominant and not numerals:
            raise ValueError("SecondaryDominant found at the end of progression")
        if progression[i] == fn.Function.SecondaryDominant:
            numerals.insert(0, f"V7/{numerals[0]}")
            continue
        numerals.insert(0, progression[i].rand_roman_numeral())
    return numerals

