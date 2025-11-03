import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"

from src.features.extract_mfcc import extract_gtzan_features
from src.features.labels import get_genre_from_label
from src.models.model1 import train_model1
from src.setup.init_dataset import ensure_gtzan_downloaded

ensure_gtzan_downloaded()
scaler, data = extract_gtzan_features()

model, history, ratios = train_model1(data)
for label, ratio in ratios.items():
    print(f"For genre {get_genre_from_label(label):<10}: Correct percentage: {100 * ratio:.4f}")
