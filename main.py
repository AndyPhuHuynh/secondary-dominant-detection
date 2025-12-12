import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"

import argparse

from src.setup.songs import NUM_DEFAULT_SONGS, setup_songs
from src.setup.soundfonts import setup_soundfonts
from src.models.logistic_regression import train_logistic_regression
from src.models.svm import train_svm
from src.features.loading import load_features
from src.visualization.box_plot import plot_mfcc_per_chord_box_plot
from src.visualization.scatter_plot import plot_mfcc_mean_vs_std_scatter_plot, plot_tonnetz_mean_scatter_plot

def main():
    parser = argparse.ArgumentParser(
        description="A program to classify weather an "
                    "eight chord long chord progression "
                    "is diatonic or contains secondary dominants")


    parser.add_argument(
        "--gen-songs",
        type=int,
        nargs="?",
        const=NUM_DEFAULT_SONGS,
        default=None,
        help="Number of songs to generate"
    )
    parser.add_argument(
        "--regen-features",
        action="store_true",
        help="Regenerate the cached features on disk"
    )
    parser.add_argument(
        "--feature-type",
        type=str,
        required=True,
        choices=[
            "global-mfcc", "per-chord-mfcc",
            "global-tonnetz", "per-chord-tonnetz",
            "hpcp", "hpcp-tonnetz"
        ],
        help="The type of features to extract from the dataset"
    )

    parser.add_argument(
        "--model",
        type=str,
        required = True,
        choices=["logistic-regression", "svm"]
    )
    args = parser.parse_args()

    setup_soundfonts()

    song_count = NUM_DEFAULT_SONGS
    force_song_setup = False
    if args.gen_songs is not None:
        song_count = args.gen_songs
        force_song_setup = True
    setup_songs(song_count, force_song_setup)

    scaler, X, y = load_features(args.feature_type, args.regen_features or force_song_setup)
    if args.feature_type == "global-mfcc":
        plot_mfcc_mean_vs_std_scatter_plot(X, y)
    elif args.feature_type == "per-chord-mfcc":
        plot_mfcc_per_chord_box_plot(X, y)
    elif args.feature_type == "global-tonnetz":
        plot_tonnetz_mean_scatter_plot(X, y)


    if args.model == "logistic-regression":
        train_logistic_regression(X, y)
    elif args.model == "svm":
        train_svm(X, y)
    else:
        raise ValueError(f"Unknown model type: {args.model}")


if __name__ == "__main__":
    main()

