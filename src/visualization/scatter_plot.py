import numpy as np
import matplotlib.pyplot as plt

import src.paths as paths
from src.features.mfcc import NUM_MFCC_STATS
from src.features.tonnetz import get_tonnetz_axis_name


def plot_mfcc_mean_vs_std_scatter_plot(X: np.ndarray, y: np.ndarray):
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


def plot_tonnetz_mean_scatter_plot(X: np.ndarray, y: np.ndarray):
    paths.GRAPHS_TONNETZ_DIR.mkdir(exist_ok=True, parents=True)

    for tonnetz_axis in range(3):
        x_index = tonnetz_axis * 2
        y_index = tonnetz_axis * 2 + 1

        x_values = X[:, x_index]
        y_values   = X[:, y_index]

        plt.figure(figsize=(10, 4))
        plt.scatter(
            x_values[y == 0],
            y_values[y == 0],
            alpha=0.4,
            label="Diatonic",
            c="C0",
            edgecolors="w"
        )

        plt.scatter(
            x_values[y == 1],
            y_values[y == 1],
            alpha=0.4,
            label="Non-diatonic",
            c="C1",
            edgecolors="w"
        )

        axis_name = get_tonnetz_axis_name(tonnetz_axis)
        plt.title(f"Tonnetz {axis_name}: Diatonic vs Secondary Dominant")
        plt.xlabel(f"{axis_name} X-coordinate (Mean)")
        plt.ylabel(f"{axis_name} Y-coordinate (Mean)")
        plt.legend(loc="best")
        plt.grid(True, alpha=0.6)

        paths.GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
        filename = paths.GRAPHS_TONNETZ_DIR / f"Global_Tonnetz_{axis_name.replace(' ', '_')}_Scatter.png"
        plt.savefig(filename)
        plt.close()
