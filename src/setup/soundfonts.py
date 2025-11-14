import os
import shutil
import urllib.request
import zipfile
import src.paths as paths

def download_soundfonts() -> None:
    if paths.FLUID_SF_PATH.exists():
        print(f"Soundfont already exists: {paths.FLUID_SF_PATH}")
        return

    os.makedirs(paths.TEMP_DIR, exist_ok=True)
    os.makedirs(paths.SOUNDFONTS_DIR, exist_ok=True)

    temp_soundfonts_path    = paths.TEMP_DIR.joinpath("soundfonts")
    zip_path                = temp_soundfonts_path.joinpath(f"{paths.FLUID_SF_NAME}.zip")
    extract_path            = temp_soundfonts_path.joinpath(f"{paths.FLUID_SF_NAME}")
    result_path             = extract_path.joinpath("FluidR3_GM.sf2")

    os.makedirs(temp_soundfonts_path, exist_ok=True)
    os.makedirs(extract_path, exist_ok=True)

    print(f"Downloading Soundfont from {paths.FLUID_SF_PATH}")
    download_url: str = "https://keymusician01.s3.amazonaws.com/FluidR3_GM.zip"
    urllib.request.urlretrieve(download_url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zip_file:
        zip_file.extractall(extract_path)
    shutil.move(result_path, paths.FLUID_SF_PATH)
    os.remove(zip_path)
    shutil.rmtree(extract_path)