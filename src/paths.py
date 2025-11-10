from pathlib import Path

PROJECT_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR:    Path = PROJECT_DIR.joinpath("data")
SAMPLES_DIR: Path = PROJECT_DIR.joinpath("samples")
HAPPY_DIR:   Path = DATA_DIR.joinpath("happy")
SAD_DIR:     Path = DATA_DIR.joinpath("sad")