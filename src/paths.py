import random
from pathlib import Path

PROJECT_DIR: Path = Path(__file__).resolve().parent.parent

DATA_DIR:       Path = PROJECT_DIR / "data"
SOUNDFONTS_DIR: Path = PROJECT_DIR / "soundfonts"
TEMP_DIR:       Path = PROJECT_DIR / "temp"
INFO_DIR:       Path = PROJECT_DIR / "info"
GRAPHS_DIR:     Path = PROJECT_DIR / "graphs"
CACHE_DIR:      Path = PROJECT_DIR / "cache"


# DATA
DATA_DIATONIC_DIR:     Path = DATA_DIR / "diatonic"
DATA_NON_DIATONIC_DIR: Path = DATA_DIR / "non-diatonic"

# SOUNDFONTS
SOUNDFONTS_CATALOG: Path = SOUNDFONTS_DIR / "catalog.json"
SOUNDFONTS_SF2_DIR: Path = SOUNDFONTS_DIR / "sf2"

# TEMP
TEMP_SOUNDFONTS_DIR: Path = TEMP_DIR / "soundfonts"
TEMP_FLUIDSYNTH_FSC: Path = TEMP_DIR / "select.fsc"

# INFO
INFO_DIATONIC_TXT:     Path = INFO_DIR / "diatonic.txt"
INFO_NON_DIATONIC_TXT: Path = INFO_DIR / "non-diatonic.txt"