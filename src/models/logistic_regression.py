import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from src.utils import split_dataset


def train_baseline(X: np.ndarray, y: np.ndarray):
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(X, y)

    model =  LogisticRegression(
        C=0.01,
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    print(f"Train accuracy: {train_acc * 100:.2f}%")
    print(f"Test accuracy: {test_acc * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, model.predict(X_test)))
    print("\nClassification Report:")
    print(classification_report(y_test, model.predict(X_test),
                                target_names=['Diatonic', 'Non-diatonic']))

    return model