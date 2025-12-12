import joblib
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

import src.paths as paths
from src.features.hpcp import HPCPExtractor, HPCPAndTonnetzExtractor
from src.features.mfcc import GlobalMFCCExtractor, PerChordMFCCExtractor
from src.features.tonnetz import GlobalTonnetzExtractor, PerChordTonnetzExtractor, TonnetzContrastExtractor


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
    return GlobalMFCCExtractor.load_features(regen_features)


def load_per_chord_mfcc(regen_features: bool):
    return PerChordMFCCExtractor.load_features(regen_features)


def load_global_tonnetz_features(regen_features: bool):
    return GlobalTonnetzExtractor.load_features(regen_features)


def load_per_chord_tonnetz_features(regen_features: bool):
    return PerChordTonnetzExtractor.load_features(regen_features)


def load_features(mode: str, regen_features: bool):
    if mode == "global-mfcc":
        return load_mfcc_features(regen_features)
    elif mode == "per-chord-mfcc":
        return load_per_chord_mfcc(regen_features)
    elif mode == "global-tonnetz":
        return load_global_tonnetz_features(regen_features)
    elif mode == "per-chord-tonnetz":
        return load_per_chord_tonnetz_features(regen_features)
    elif mode == "tonnetz-contrast":
        return TonnetzContrastExtractor.load_features(regen_features)
    elif mode == "hpcp":
        return HPCPExtractor.load_features(regen_features)
    elif mode == "hpcp-tonnetz":
        return HPCPAndTonnetzExtractor.load_features(regen_features)
    else:
        raise ValueError(f"Invalid feature mode: {mode}")
