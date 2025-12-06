import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"

from src.setup.soundfonts import setup_soundfonts

from src.features.mfcc import extract_mfcc_from_dataset
from src.models.model1 import train_model1
from src.music.song import generate_songs
from src.visualization.scatter_plot import plot_mfcc_mean_vs_std


def run_iteration():
    generate_songs()
    scaler, X, y = extract_mfcc_from_dataset(44100)
    model, history, ratios = train_model1(X, y)
    for i in range(1, 14):
        plot_mfcc_mean_vs_std(X, y, i)
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
        print(f"    Diatonic:     {ratio_iterations[i][0]}")
        print(f"    Non-diatonic: {ratio_iterations[i][1]}")

    print(f"\nDiatonic average:     {float(diatonic_sum)/num_iterations}")
    print(  f"Non-diatonic average: {float(non_diatonic_sum)/num_iterations}")



if __name__ == "__main__":
    setup_soundfonts()
    run_iteration()
