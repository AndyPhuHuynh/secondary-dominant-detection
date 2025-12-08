import json
import random
import src.paths as paths

_catalog = None
_HARMONIC_INSTRUMENT_RANGES = [
    (0, 7),      # Piano
    (8, 15),     # Chromatic Percussion (vibes, marimba, etc.)
    (16, 23),    # Organ
    (24, 31),    # Guitar
    (32, 39),    # Bass
    (41, 47),    # Strings
    (48, 55),    # Ensemble
    (56, 63),    # Brass
    (64, 71),    # Reed
    (72, 79),    # Pipe
    (80, 87),    # Synth Lead
    (88, 95),    # Synth Pad
    (96, 103),   # Synth Effects
    (104, 111),  # Ethnic
    # Skip 112-119 (Percussive/Sound Effects)
    # Skip 120-127 (Sound Effects)
]


def _is_harmonic_preset(preset: int) -> bool:
    for range_ in _HARMONIC_INSTRUMENT_RANGES:
        if range_[0] <= preset <= range_[1]:
            return True
    return False


def _load_catalog():
    global _catalog
    if _catalog is not None:
        return _catalog

    if not paths.SOUNDFONTS_CATALOG.exists():
        raise FileNotFoundError(f'{paths.SOUNDFONTS_CATALOG} not found. No catalog to load')

    with paths.SOUNDFONTS_CATALOG.open() as f:
        _catalog = json.load(f)

    return _catalog


def get_random_soundfont_preset():
    """
    Get a random preset from the catalog
    :return:
    """
    catalog = _load_catalog()
    soundfont_name = random.choice(list(catalog.keys()))
    soundfont_info = catalog[soundfont_name]

    all_presets = list(soundfont_info["presets"])
    all_presets = [item for item in all_presets if _is_harmonic_preset(item["preset"])]

    if not all_presets:
        raise ValueError("No presets found")

    preset = random.choice(all_presets)
    return {
        "path": soundfont_info["path"],
        "name": preset["name"],
        "bank": preset["bank"],
        "preset": preset["preset"],
    }




