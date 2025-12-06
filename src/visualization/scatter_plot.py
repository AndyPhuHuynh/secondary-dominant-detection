import numpy as np
import matplotlib.pyplot as plt

import src.paths as paths
from src.features.labels import get_label_num_to_string


def plot_mfcc_mean_vs_std(X: np.ndarray, y: np.ndarray, mfcc_index: int = 1):
    mean_col_index = (mfcc_index - 1) * 4
    std_col_index = mean_col_index + 1

    mfcc_mean = X[:,mean_col_index]
    mfcc_std  = X[:,std_col_index]

    plt.figure(figsize=(10, 4))
    plt.scatter(
        mfcc_mean[y == 0],
        mfcc_std[y == 0],
        alpha=0.6,
        label=get_label_num_to_string(0),
        c="C0",
        edgecolors="w"
    )

    plt.scatter(
        mfcc_mean[y == 1],
        mfcc_std[y == 1],
        alpha=0.6,
        label=get_label_num_to_string(1),
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
    filename = paths.GRAPHS_DIR / f"scatter_{feature_name}_mean_vs_std.png"
    plt.savefig(filename)
    plt.close()
