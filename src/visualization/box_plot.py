import matplotlib.pyplot as plt
import numpy as np

import src.constants as c
import src.paths as paths
from src.features.mfcc import mfcc_feature_index, mfcc_stat_index_to_str


def plot_mfcc_per_chord_box_plot(X, y, mfcc_index, stat_index, show_plot: bool):
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