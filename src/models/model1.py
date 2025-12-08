import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical

from src.utils import split_dataset


def evaluate_predictions(model, test_ds, y_test, num_classes):
    y_predictions = np.argmax(model.predict(test_ds, verbose=0), axis=1)
    ratios: dict[int, float] = {}
    for i in range(num_classes):
        mask = y_test == i
        total = np.sum(mask)
        correct = np.sum(y_predictions[mask] == i)
        ratios[i] = correct / total if total > 0 else 0
    return ratios


def train_model1(X: np.ndarray, y: np.ndarray):
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(X, y)

    num_features = X.shape[1]
    num_classes = 2

    y_train_cat      = to_categorical(y_train, num_classes)
    y_validation_cat = to_categorical(y_val,   num_classes)
    y_test_cat       = to_categorical(y_test,  num_classes)

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train_cat))
    val_ds   = tf.data.Dataset.from_tensor_slices((X_val,   y_validation_cat))
    test_ds  = tf.data.Dataset.from_tensor_slices((X_test,  y_test_cat))

    batch_size = 32
    train_ds = train_ds.shuffle(2048).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    model = Sequential([
        Input(shape=(num_features,)),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=64, batch_size=32,
        callbacks=[early_stop]
    )

    test_loss, test_acc = model.evaluate(test_ds, verbose=2)

    ratios: dict[int, float] = evaluate_predictions(model, test_ds, y_test, num_classes)

    print(f"Training accuracy: {history.history["accuracy"][-1]:.2f}")
    print(f"Test accuracy: {test_acc * 100:.2f}%")
    print(f"Test loss: {test_loss:.4f}")
    return model, history, ratios


def train_model2(X: np.ndarray, y: np.ndarray):
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(X, y)

    num_features = X.shape[1]
    num_classes = 2

    y_train_cat      = to_categorical(y_train, num_classes)
    y_validation_cat = to_categorical(y_val,   num_classes)
    y_test_cat       = to_categorical(y_test,  num_classes)

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train_cat))
    val_ds   = tf.data.Dataset.from_tensor_slices((X_val,   y_validation_cat))
    test_ds  = tf.data.Dataset.from_tensor_slices((X_test,  y_test_cat))

    batch_size = 32
    train_ds = train_ds.shuffle(2048).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    model = Sequential([
        Input(shape=(num_features,)),
        Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20, batch_size=32,
        # callbacks=[early_stop]
    )

    test_loss, test_acc = model.evaluate(test_ds, verbose=2)

    ratios: dict[int, float] = evaluate_predictions(model, test_ds, y_test, num_classes)

    print(f"Training accuracy: {history.history["accuracy"][-1]:.2f}")
    print(f"Test accuracy: {test_acc * 100:.2f}%")
    print(f"Test loss: {test_loss:.4f}")
    return model, history, ratios
