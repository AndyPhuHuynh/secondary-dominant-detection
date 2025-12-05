import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"

import src.visualization.histogram as plots
from src.setup.soundfonts import setup_soundfonts

from src.features.labels import get_label_num_to_string
from src.features.mfcc import extract_mfcc_from_dataset
from src.models.model1 import train_model1
from src.music.song import generate_songs

setup_soundfonts()
generate_songs()
scaler, X, y = extract_mfcc_from_dataset(44100)

model, history, ratios = train_model1(X, y)
for effect, accuracy in ratios.items():
    print(f"Accuracy for {get_label_num_to_string(effect):<10} is {100 * accuracy:.2f}%")