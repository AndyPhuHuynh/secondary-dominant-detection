import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.features.labels import get_label_num_to_string

def plot_mfcc_distributions(X, y, figsize=(14, 6)):
    stats = ["mean", "std", "min", "max"]

    for stat_index, stat_name in enumerate(stats):
        plot_data = []

        for mfcc_num in range(13):
            feature_index = mfcc_num * 4 + stat_index

            for label in [0, 1]:
                mask = (y == label)
                values = X[mask, feature_index]

                for val in values:
                    plot_data.append({
                        "MFCC": f"MFCC{mfcc_num+1}",
                        "Value": val,
                        "Label": get_label_num_to_string(label)
                    })

        df = pd.DataFrame(plot_data)

        # create a separate, larger figure for this stat
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        sns.boxplot(data=df, x="MFCC", y="Value", hue="Label", ax=ax)
        ax.set_title(f"MFCC {stat_name}")
        ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.show()


def plot_mfcc_profiles(X, y):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    stats = ["mean", "std", "min", "max"]

    for stat_index, (ax, stat_name) in enumerate(zip(axes.flat, stats)):
        mfcc_values_label0 = []
        mfcc_values_label1 = []

        for mfcc_num in range(13):
            feature_index = mfcc_num * 4 + stat_index
            mfcc_values_label0.append(X[y == 0, feature_index].mean())
            mfcc_values_label1.append(X[y == 1, feature_index].mean())

        mfcc_numbers = list(range(1, 14))
        ax.plot(mfcc_numbers, mfcc_values_label0, marker="o",
                label=get_label_num_to_string(0), linewidth=2)
        ax.plot(mfcc_numbers, mfcc_values_label1, marker="o",
                label=get_label_num_to_string(1), linewidth=2)

        ax.set_xlabel("MFCC Coefficient")
        ax.set_ylabel(f"{stat_name} Value")
        ax.set_title(f"Average MFCC {stat_name} Profile")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_mean_and_effects(X, y, stat_index):
    """
    Plot class means +/- std for each MFCC coefficient (stat index 0..3),
    plus mean difference and Cohen's d per coefficient.
    Assumes features are arranged as MFCC1_(mean,std,min,max), MFCC2_..., etc.
    """
    stats = ["mean", "std", "min", "max"]
    n_mfcc = 13
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [2, 1, 1]})
    ax_top, ax_diff, ax_d = axes

    # choose which aggregated stat to visualize (use mean here)
    mfcc_nums = np.arange(1, n_mfcc + 1)

    means0 = np.array([X[y == 0, i*4 + stat_index].mean() for i in range(n_mfcc)])
    std0   = np.array([X[y == 0, i*4 + stat_index].std()  for i in range(n_mfcc)])
    means1 = np.array([X[y == 1, i*4 + stat_index].mean() for i in range(n_mfcc)])
    std1   = np.array([X[y == 1, i*4 + stat_index].std()  for i in range(n_mfcc)])

    # top: means with shaded std
    ax_top.plot(mfcc_nums, means0, marker="o", label=get_label_num_to_string(0), linewidth=2)
    ax_top.fill_between(mfcc_nums, means0 - std0, means0 + std0, alpha=0.2)
    ax_top.plot(mfcc_nums, means1, marker="o", label=get_label_num_to_string(1), linewidth=2)
    ax_top.fill_between(mfcc_nums, means1 - std1, means1 + std1, alpha=0.2)
    ax_top.set_title(f"Class means (stat {stats[stat_index]}) with ±1 std")
    ax_top.set_xlabel("MFCC coefficient")
    ax_top.set_ylabel("Value")
    ax_top.legend()
    ax_top.grid(alpha=0.3)

    # middle: mean difference
    diff = means1 - means0
    ax_diff.bar(mfcc_nums, diff, color="C2", alpha=0.7)
    ax_diff.axhline(0, color="k", linewidth=0.7)
    ax_diff.set_title("Mean difference (label1 - label0)")
    ax_diff.set_xlabel("MFCC coefficient")
    ax_diff.set_ylabel("Difference")
    ax_diff.grid(alpha=0.3)

    # bottom: Cohen's d (effect size)
    pooled_std = np.sqrt(((std0 ** 2) + (std1 ** 2)) / 2)
    # avoid div by zero
    pooled_std[pooled_std == 0] = 1e-8
    cohens_d = diff / pooled_std
    ax_d.bar(mfcc_nums, cohens_d, color="C3", alpha=0.7)
    ax_d.axhline(0, color="k", linewidth=0.7)
    ax_d.set_title("Cohen's d (approx.) per coefficient")
    ax_d.set_xlabel("MFCC coefficient")
    ax_d.set_ylabel("Cohen's d")
    ax_d.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()