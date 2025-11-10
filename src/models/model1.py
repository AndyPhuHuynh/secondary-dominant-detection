import numpy as np
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

from src.dataset import Dataset
from src.features.mfcc import split_dataset

def train_model1(dataset: Dataset):
    train, validation, test = split_dataset(dataset)

    num_features = train.X.shape[1]
    num_classes = 2

    y_train_cat      = to_categorical(train.y,      num_classes)
    y_validation_cat = to_categorical(validation.y, num_classes)
    y_test_cat       = to_categorical(test.y,       num_classes)

    model = Sequential([
        Input(shape=(num_features,)),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train.X, y_train_cat, validation_data=(validation.X, y_validation_cat), epochs=32, batch_size=32)
    test_loss, test_acc = model.evaluate(test.X, y_test_cat, verbose=2)

    ratios: dict[int, float] = {}
    y_predictions = np.argmax(model.predict(test.X, verbose=0), axis=1)
    for i in range(num_classes):
        mask = test.y == i
        total = np.sum(mask)
        correct = np.sum(y_predictions[mask] == i)
        ratios[i] = correct / total if total > 0 else 0

    print(f"Test accuracy: {test_acc * 100:.2f}%")
    print(f"Test loss: {test_loss:.4f}")
    return model, history, ratios

