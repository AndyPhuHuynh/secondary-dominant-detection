import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical

from src.utils import split_dataset


def build_model1(num_features, num_classes):
    model = Sequential([
        Input(shape=(num_features,)),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def build_model2(num_features, num_classes):
    model = Sequential([
        Input(shape=(num_features,)),
        Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def plot_learning_curve_model2(X, y):
    shuffle_indices = np.random.permutation(len(X))
    X = X[shuffle_indices]
    y = y[shuffle_indices]

    train_sizes = np.linspace(0.1, 1.0, 10)
    num_features = X.shape[1]
    num_classes = 2

    train_accs = []
    val_accs = []

    for train_size in train_sizes:
        np.random.seed(0)
        tf.random.set_seed(0)

        subset_size = int(len(y) * train_size)

        X_subset = X[:subset_size]
        y_subset = y[:subset_size]

        X_train, X_val, y_train, y_val = train_test_split(
            X_subset, y_subset, test_size=0.2, random_state=0, stratify=y_subset
        )

        y_train_cat = to_categorical(y_train, num_classes)
        y_val_cat = to_categorical(y_val, num_classes)

        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train_cat))
        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val_cat))

        train_ds = train_ds.shuffle(2048).batch(32).prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.batch(32).prefetch(tf.data.AUTOTUNE)

        model = build_model2(num_features, num_classes)
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True
        )
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=64,
            callbacks=[early_stop]
        )

        train_loss, train_acc = model.evaluate(train_ds, verbose=0)
        val_loss, val_acc = model.evaluate(val_ds, verbose=0)

        train_accs.append(train_acc)
        val_accs.append(val_acc)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_accs, "o-", label="Training accuracy")
    plt.plot(train_sizes, val_accs, "o-", label="Validation accuracy")
    plt.title("Learning Curve for Neural Network")
    plt.xlabel(f"Training set size (fraction of {len(y)})")
    plt.grid(True)
    plt.legend()
    plt.show()


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

    model = build_model2(num_features, num_classes)

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

    model = build_model2(num_features, num_classes)

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20, batch_size=32,
        callbacks=[early_stop]
    )

    test_loss, test_acc = model.evaluate(test_ds, verbose=2)

    ratios: dict[int, float] = evaluate_predictions(model, test_ds, y_test, num_classes)

    print(f"Training accuracy: {history.history["accuracy"][-1]:.2f}")
    print(f"Test accuracy: {test_acc * 100:.2f}%")
    print(f"Test loss: {test_loss:.4f}")
    return model, history, ratios
