import joblib
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

import src.paths as paths
from src.features.chroma import extract_stft_from_dataset
from src.features.mfcc import  extract_mfcc_from_dataset


def _load_feature(
        extract_fn,
        feature_cache_path: Path,
        scaler_cache_path: Path,
        regen_features: bool,
):
    if not feature_cache_path.exists() or not scaler_cache_path.exists() or regen_features:
        print("Extracting feature set")
        X_unscaled, y = extract_fn(44100)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_unscaled)

        paths.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(feature_cache_path, X=X_scaled, y=y)
        joblib.dump(scaler, scaler_cache_path)

        return scaler, X_scaled, y

    print(f"Loading cached features from {feature_cache_path}")
    loaded_data = np.load(feature_cache_path)
    X_loaded = np.array(loaded_data["X"])
    y_loaded = np.array(loaded_data["y"])
    scaler_loaded = joblib.load(scaler_cache_path)

    return scaler_loaded, X_loaded, y_loaded


def load_mfcc_features(regen_features: bool):
    return _load_feature(
        extract_mfcc_from_dataset,
        feature_cache_path = paths.CACHE_DIR / "mfcc_features.npz",
        scaler_cache_path = paths.CACHE_DIR / "mfcc_scaler.pkl",
        regen_features = regen_features
    )


def load_stft_features(regen_features: bool):
    return _load_feature(
        extract_stft_from_dataset,
        feature_cache_path=paths.CACHE_DIR / "stft_features.npz",
        scaler_cache_path=paths.CACHE_DIR / "stft_scaler.pkl",
        regen_features=regen_features
    )


def load_features(mode: str, regen_features: bool):
    if mode == "stft":
        return load_stft_features(regen_features)
    elif mode == "mfcc":
        return load_mfcc_features(regen_features)
    else:
        raise ValueError(f"Invalid feature mode: {mode}")
