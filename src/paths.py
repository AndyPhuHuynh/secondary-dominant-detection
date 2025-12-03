from pathlib import Path

PROJECT_DIR: Path = Path(__file__).resolve().parent.parent

DATA_DIR:       Path = PROJECT_DIR / "data"
SAMPLES_DIR:    Path = PROJECT_DIR / "samples"
SOUNDFONTS_DIR: Path = PROJECT_DIR / "soundfonts"
TEMP_DIR:       Path = PROJECT_DIR / "temp"
INFO_DIR:       Path = PROJECT_DIR / "info"

# SOUNDFONTS
FLUID_SF_NAME: str = "FluidR3_GM"
FLUID_SF_PATH: Path = SOUNDFONTS_DIR / f"{FLUID_SF_NAME}.sf2"

# DATA
DIATONIC_DATA_DIR:     Path = DATA_DIR / "diatonic"
NON_DIATONIC_DATA_DIR: Path = DATA_DIR / "non-diatonic"


# INFO
DIATONIC_INFO:     Path = INFO_DIR / "diatonic.txt"
NON_DIATONIC_INFO: Path = INFO_DIR / "non-diatonic.txt"
