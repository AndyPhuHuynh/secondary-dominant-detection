import librosa
import matplotlib.pyplot as plt

def plot_mfcc(fig, axes, path, index, title):
    signal, sr = librosa.load(path, sr=22050)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)

    # Plot MFCC on the corresponding axis
    img = librosa.display.specshow(mfcc, x_axis="time", sr=sr, ax=axes[index])
    axes[index].set_title(title)
    axes[index].label_outer()  # Hide inner labels to reduce clutter
    fig.colorbar(img, ax=axes[index], format="%+2.0f dB")


def plot_examples_one_each():
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))  # 2 rows × 5 columns
    axes = axes.flatten()  # make it a 1D list for easy indexing

    plot_mfcc(fig, axes, "data/samples/dry/dry_sine_000.wav",0, "Dry 0")
    plot_mfcc(fig, axes, "data/samples/dry/dry_square_001.wav",1, "Dry 1")
    plot_mfcc(fig, axes, "data/samples/dry/dry_triangle_002.wav",2, "Dry 2")
    plot_mfcc(fig, axes, "data/samples/reverb/reverb_sine_000.wav",3, "Reverb 0")
    plot_mfcc(fig, axes, "data/samples/reverb/reverb_square_001.wav",4, "Reverb 1")
    plot_mfcc(fig, axes, "data/samples/reverb/reverb_triangle_002.wav",5, "Reverb 2")
    plot_mfcc(fig, axes, "data/samples/distortion/distortion_sine_000.wav", 6, "Distortion 0")
    plot_mfcc(fig, axes, "data/samples/distortion/distortion_square_001.wav", 7, "Distortion 1")
    plot_mfcc(fig, axes, "data/samples/distortion/distortion_triangle_002.wav", 8, "Distortion 2")

    plt.tight_layout(pad=3.0)
    plt.show()


def plot_10_pop_examples():
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))  # 2 rows × 5 columns
    axes = axes.flatten()  # make it a 1D list for easy indexing

    plot_mfcc(fig, axes, "data/GTZAN/pop/pop.00001.au", 0, "Pop 1")
    plot_mfcc(fig, axes, "data/GTZAN/pop/pop.00002.au", 1, "Pop 2")
    plot_mfcc(fig, axes, "data/GTZAN/pop/pop.00003.au", 2, "Pop 3")
    plot_mfcc(fig, axes, "data/GTZAN/pop/pop.00004.au", 3, "Pop 4")
    plot_mfcc(fig, axes, "data/GTZAN/pop/pop.00005.au", 4, "Pop 5")
    plot_mfcc(fig, axes, "data/GTZAN/pop/pop.00006.au", 5, "Pop 6")
    plot_mfcc(fig, axes, "data/GTZAN/pop/pop.00007.au", 6, "Pop 7")
    plot_mfcc(fig, axes, "data/GTZAN/pop/pop.00008.au", 7, "Pop 8")
    plot_mfcc(fig, axes, "data/GTZAN/pop/pop.00009.au", 8, "Pop 9")
    plot_mfcc(fig, axes, "data/GTZAN/pop/pop.00010.au", 9, "Pop 10")

    plt.tight_layout(pad=3.0)
    plt.show()


