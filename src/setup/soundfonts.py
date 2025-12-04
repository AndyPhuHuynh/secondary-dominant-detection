import py7zr
import shutil
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import src.paths as paths

DOWNLOAD_URLS: dict[str, str] = {
    "FluidR3_GM": "https://keymusician01.s3.amazonaws.com/FluidR3_GM.zip",
    "RhodesLA":   "https://musical-artifacts.com/artifacts/7017/Rhodes_LA.7z",
}

def unzip(zip_path: Path, extract_path: Path):
    if zip_path.suffix == ".zip":
        with zipfile.ZipFile(zip_path, "r") as zip_file:
            zip_file.extractall(extract_path)
    elif zip_path.suffix == ".7z":
        with py7zr.SevenZipFile(zip_path, "r") as zip_file:
            zip_file.extractall(extract_path)
    else:
        raise ValueError(f"Unsupported zip format: {zip_path}")


def download_soundfonts() -> None:
    paths.TEMP_SOUNDFONTS_DIR.mkdir(parents=True, exist_ok=True)
    paths.SOUNDFONTS_DIR.mkdir(parents=True, exist_ok=True)

    for name, download_url in DOWNLOAD_URLS.items():
        dest_path = paths.SOUNDFONTS_DIR / f"{name}.sf2"
        if dest_path.exists():
            print(f"Soundfont already exists: {dest_path}")
            continue

        parsed_url = urllib.parse.urlparse(download_url)
        file_extension = Path(parsed_url.path).suffix

        zip_path     = paths.TEMP_SOUNDFONTS_DIR / f"{name}{file_extension}"
        extract_path = paths.TEMP_SOUNDFONTS_DIR / name

        extract_path.mkdir(parents=True, exist_ok=True)
        print(f"Downloading Soundfont from {download_url}")

        req = urllib.request.Request(
            download_url,
            headers={
                "User-Agent": "PythonSoundfontDownloader/1.0"
            }
        )

        with urllib.request.urlopen(req) as response, open(zip_path, "wb") as my_file:
            my_file.write(response.read())

        unzip(zip_path, extract_path)
        for file in extract_path.iterdir():
            if not file.suffix == ".sf2": continue
            shutil.move(file, dest_path)

            print(f"File: {file}")
            print(f"Moved: {dest_path}")

        zip_path.unlink()
        shutil.rmtree(extract_path)
    shutil.rmtree(paths.TEMP_DIR)