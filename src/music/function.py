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
    Function.SecondaryDominant: [Function.Tonic, Function.TonicLike, Function.Subdominant, Function.Dominant],
}