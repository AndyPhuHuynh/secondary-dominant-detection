import joblib
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import src.paths as paths
from src.features.labels import get_label_string_to_num


class FeatureExtractor(ABC):
    FEATURE_NAME: str = "base"
    FEATURE_CACHE_PATH: Path
    SCALER_CACHE_PATH: Path

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.FEATURE_NAME == "base":
            raise TypeError(f"{cls.__name__} must define FEATURE_NAME")

        cls.FEATURE_CACHE_PATH = paths.CACHE_DIR / f"{cls.FEATURE_NAME}-features.npz"
        cls.SCALER_CACHE_PATH = paths.CACHE_DIR / f"{cls.FEATURE_NAME}-scaler.pkl"


    @classmethod
    @abstractmethod
    def extract_features_from_file(cls, filepath: Path):
        raise NotImplementedError


    @classmethod
    def extract_features_from_dataset(cls):
        X = []
        y = []

        for path in paths.DATA_DIR.iterdir():
            if not path.is_dir():
                continue

            label_str = path.name
            label_num: int
            try:
                label_num = get_label_string_to_num(label_str)
            except Exception as e:
                print(f"Invalid label found within data folder: '{label_str}': {e}")
                continue

            for filepath in tqdm(list(path.iterdir()), desc=f"Extracting {cls.FEATURE_NAME} for {label_str:<15}"):
                try:
                    features = cls.extract_features_from_file(filepath)
                    if features is not None:
                        X.append(features)
                        y.append(label_num)
                except Exception as e:
                    print(f"Unable to process file '{filepath}': {e}")

        return np.array(X), np.array(y)


    @classmethod
    def load_features(cls, regen_features: bool):
        if not cls.FEATURE_CACHE_PATH.exists() or not cls.SCALER_CACHE_PATH.exists() or regen_features:
            print("Extracting feature set")
            X_unscaled, y = cls.extract_features_from_dataset()

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_unscaled)

            paths.CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cls.FEATURE_CACHE_PATH, X=X_scaled, y=y)
            joblib.dump(scaler, cls.SCALER_CACHE_PATH)

            return scaler, X_scaled, y

        print(f"Loading cached features from {cls.FEATURE_CACHE_PATH} and scaler from {cls.SCALER_CACHE_PATH}")
        loaded_data = np.load(cls.FEATURE_CACHE_PATH)
        X_loaded = np.array(loaded_data["X"])
        y_loaded = np.array(loaded_data["y"])
        scaler_loaded = joblib.load(cls.SCALER_CACHE_PATH)

        return scaler_loaded, X_loaded, y_loaded