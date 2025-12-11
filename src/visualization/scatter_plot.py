import numpy as np
import matplotlib.pyplot as plt

import src.paths as paths


def plot_mfcc_mean_vs_std(X: np.ndarray, y: np.ndarray):
    mfcc_graph_paths = paths.GRAPHS_DIR / "mfcc"
    mfcc_graph_paths.mkdir(exist_ok=True, parents=True)

    for mfcc_index in range(1, 14):
        mean_col_index = (mfcc_index - 1) * 4
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
        filename = mfcc_graph_paths / f"{feature_name}_mean_vs_std.png"
        plt.savefig(filename)
        plt.close()
