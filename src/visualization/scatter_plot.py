import numpy as np
import matplotlib.pyplot as plt

import src.constants as c
import src.paths as paths
from src.features.mfcc import NUM_MFCC_STATS, mfcc_feature_index, mfcc_stat_index_to_str


def plot_mfcc_mean_vs_std(X: np.ndarray, y: np.ndarray):
    paths.GRAPHS_MFCC_DIR.mkdir(exist_ok=True, parents=True)

    for mfcc_index in range(1, 14):
        mean_col_index = (mfcc_index - 1) * NUM_MFCC_STATS
        std_col_index = mean_col_index + 1

        mfcc_mean = X[:, mean_col_index]
        mfcc_std = X[:, std_col_index]

        plt.figure(figsize=(10, 4))
        plt.scatter(
            mfcc_mean[y == 0],
            mfcc_std[y == 0],
            alpha=0.4,
            label="Diatonic",
            c="C0",
            edgecolors="w"
        )

        plt.scatter(
            mfcc_mean[y == 1],
            mfcc_std[y == 1],
            alpha=0.4,
            label="Non-diatonic",
            c="C1",
            edgecolors="w"
        )

        feature_name = f"MFCC{mfcc_index}"
        plt.title(f"Scatter plot: {feature_name} mean vs. {feature_name} std")
        plt.xlabel(f"{feature_name} Mean Feature Value")
        plt.ylabel(f"{feature_name} Standard Deviation")
        plt.legend(loc="best")
        plt.grid(True, alpha=0.6)

        paths.GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
        filename = paths.GRAPHS_MFCC_DIR / f"mfcc{mfcc_index}_mean_vs_std.png"
        plt.savefig(filename)
        plt.close()


def plot_mfcc_box(X, y, mfcc_index, stat_index, show_plot: bool):
    positions_diatonic = np.arange(1, c.NUM_CHORDS + 1) - 0.2
    positions_non_diatonic = np.arange(1, c.NUM_CHORDS + 1) + 0.2

    plt.figure(figsize=(12, 6))

    X_diatonic = X[y == 0]
    X_non_diatonic = X[y == 1]

    diatonic_data = []
    non_diatonic_data = []

    for chord in range(c.NUM_CHORDS):
        col = mfcc_feature_index(mfcc_index, chord, stat_index)

        diatonic_data.append(X_diatonic[:, col])
        non_diatonic_data.append(X_non_diatonic[:, col])

    bp1 = plt.boxplot(
        diatonic_data,
        positions=positions_diatonic,
        widths=0.3,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor="tab:blue"),
        medianprops=dict(color="black")
    )

    bp2 = plt.boxplot(
        non_diatonic_data,
        positions=positions_non_diatonic,
        widths=0.3,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor="tab:orange"),
        medianprops=dict(color="black")
    )

    chord_positions = range(1, c.NUM_CHORDS + 1)
    plt.xticks(
        ticks=chord_positions,
        labels=[str(i) for i in chord_positions]
    )
    plt.xlabel("Chord Index")
    plt.ylabel(f"MFCC{mfcc_index+1} {mfcc_stat_index_to_str(stat_index)}")
    plt.title(f"Per-Chord Distribution of MFCC{mfcc_index+1} {mfcc_stat_index_to_str(stat_index)}")

    plt.legend(
        [bp1["boxes"][0], bp2["boxes"][0]],
        ["Diatonic", "Non-diatonic"]
    )

    plt.tight_layout()

    if show_plot:
        plt.show()
    else:
        paths.GRAPHS_MFCC_DIR.mkdir(exist_ok=True, parents=True)
        filename = paths.GRAPHS_MFCC_DIR / f"per-chord_distribution_of_mfcc{mfcc_index+1}_{mfcc_stat_index_to_str(stat_index)}_box_plot"
        plt.savefig(filename)
        plt.close()
