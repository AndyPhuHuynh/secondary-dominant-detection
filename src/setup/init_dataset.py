import os
import kagglehub
import shutil

from src.paths import GTZAN_DATA_DIR

def ensure_gtzan_downloaded() -> bool:
    if os.path.exists(GTZAN_DATA_DIR):
        print("GTZAN already downloaded")
        return True
    print("Downloading GTZAN dataset from Kagglehub")
    dataset = "carlthome/gtzan-genre-collection"

    try:
        cache_path = kagglehub.dataset_download(dataset)
        print(f"GTZAN downloaded to {cache_path}")

        genres_path: str = os.path.join(cache_path, "genres")
        if not os.path.isdir(genres_path):
            possible_paths = [
                os.path.join(cache_path, d, "genres")
                for d in os.listdir(cache_path)
                if os.path.isdir(os.path.join(cache_path, d, "genres"))
            ]
            if possible_paths:
                genres_path = possible_paths[0]

        if not os.path.isdir(genres_path):
            raise FileNotFoundError(f"Could not locate 'genres' folder in {cache_path}")

        os.makedirs(GTZAN_DATA_DIR, exist_ok=True)
        shutil.copytree(genres_path, GTZAN_DATA_DIR, dirs_exist_ok=True)
        print(f"GTZAN dataset copied to {GTZAN_DATA_DIR}")

        return True
    except Exception as e:
        print(f"Failed to download GTZAN dataset: {e}")
        print("Please download it manually from Kaggle and unzip the contents into \"data/GTZAN\":")
        print("  https://www.kaggle.com/datasets/carlthome/gtzan-genre-collection")
        return False