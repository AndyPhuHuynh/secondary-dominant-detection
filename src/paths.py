import random
from pathlib import Path

PROJECT_DIR: Path = Path(__file__).resolve().parent.parent

DATA_DIR:       Path = PROJECT_DIR / "data"
SOUNDFONTS_DIR: Path = PROJECT_DIR / "soundfonts"
TEMP_DIR:       Path = PROJECT_DIR / "temp"
INFO_DIR:       Path = PROJECT_DIR / "info"


# DATA
DATA_DIATONIC_DIR:     Path = DATA_DIR / "diatonic"
DATA_NON_DIATONIC_DIR: Path = DATA_DIR / "non-diatonic"

# TEMP
TEMP_SOUNDFONTS_DIR: Path = TEMP_DIR / "soundfonts"

# INFO
DIATONIC_INFO:     Path = INFO_DIR / "diatonic.txt"
NON_DIATONIC_INFO: Path = INFO_DIR / "non-diatonic.txt"


_soundfont_cache: list[Path] | None = None


def get_random_soundfont_path(force_refresh: bool = False) -> Path:
    global _soundfont_cache
    if _soundfont_cache is None or force_refresh:
        _soundfont_cache = list(SOUNDFONTS_DIR.glob("*.sf2"))

    if not _soundfont_cache:
        raise FileNotFoundError(f"No .sf2 files found in {SOUNDFONTS_DIR}")

    return random.choice(_soundfont_cache)