import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"

import argparse
import joblib
import numpy as np


from sklearn.preprocessing import StandardScaler

import src.paths as paths
from src.models.logistic_regression import train_baseline
from src.setup.soundfonts import setup_soundfonts
from src.music.song import generate_songs
from src.models.model1 import train_model1
from src.features.chroma import extract_stft_from_dataset
from src.features.loading import load_features


def run_iteration():
    # scaler, X, y = extract_mfcc_from_dataset(44100)
    scaler, X, y = extract_stft_from_dataset(44100)
    model, history, ratios = train_model1(X, y)
    return ratios


def run_iterations():
    ratio_iterations = []
    num_iterations = 1
    for i in range(num_iterations):
        ratio = run_iteration()
        ratio_iterations.append(ratio)

    diatonic_sum = 0
    non_diatonic_sum = 0
    for i in range(num_iterations):
        diatonic_sum += ratio_iterations[i][0]
        non_diatonic_sum += ratio_iterations[i][1]

        print(f"Iteration {i:>3}")
        print(f"    Diatonic:     {ratio_iterations[i][0]:.2f}")
        print(f"    Non-diatonic: {ratio_iterations[i][1]:.2f}")

    print(f"\nDiatonic average:     {float(diatonic_sum)/num_iterations:.2f}")
    print(  f"Non-diatonic average: {float(non_diatonic_sum)/num_iterations:.2f}")




def main():
    parser = argparse.ArgumentParser(
        description="A program to classify weather an "
                    "eight chord long chord progression "
                    "is diatonic or contains secondary dominants")

    parser.add_argument(
        "--regen-features",
        action="store_true",
        help="Regenerate the cached features on disk"
    )

    parser.add_argument(
        "--feature-type",
        type=str,
        required=True,
        choices=["mfcc", "stft"],
        help="The type of features to extract from the dataset"
    )


    args = parser.parse_args()

    setup_soundfonts()
    scaler, X, y = load_features(args.feature_type, args.regen_features)
    train_baseline(X, y)


if __name__ == "__main__":
    main()
