import subprocess
from pathlib import Path

import src.paths as paths

def midi_to_wave(
    midi_path: Path,
    wave_path: Path,
    soundfont_path: Path,
    bank: int = 0,
    preset: int = 0
):
    paths.TEMP_FLUIDSYNTH_FSC.parent.mkdir(parents=True, exist_ok=True)
    paths.TEMP_FLUIDSYNTH_FSC.write_text(f"select 0 1 {bank} {preset}")

    subprocess.run([
        "fluidsynth", "-ni",
        "-F", wave_path,
        "-r", "44100",
        "-f", paths.TEMP_FLUIDSYNTH_FSC,
        soundfont_path,
        midi_path,
    ], check=True, stdout=subprocess.DEVNULL)
