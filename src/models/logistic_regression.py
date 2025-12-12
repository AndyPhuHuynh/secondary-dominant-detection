import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.features.utils import evaluate_precision_and_recall
from src.visualization.learning_curve import plot_learning_curve
from src.visualization.roc import plot_roc_curve
from src.utils import split_dataset


def tune_hyperparameters(X_train, y_train, X_val, y_val):
    C_values = np.logspace(-4, 4, 10)

    train_scores = []
    val_scores = []

    for C in C_values:
        model = LogisticRegression(C=C, max_iter=1000, random_state=42)
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

        train_scores.append(accuracy_score(y_train, train_pred))
        val_scores.append(accuracy_score(y_val, val_pred))

    # Find best C value based on validation accuracy
    best_idx = np.argmax(val_scores)
    best_C = C_values[best_idx]
    best_val_acc = val_scores[best_idx]

    print(f"Best C value: {best_C:.4f}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.semilogx(C_values, train_scores, label='Training Accuracy', marker='o', markersize=3)
    plt.semilogx(C_values, val_scores, label='Validation Accuracy', marker='o', markersize=3)
    plt.axvline(best_C, color='red', linestyle='--', label=f'Best C = {best_C:.4f}')
    plt.xlabel('C (Regularization Parameter)')
    plt.ylabel('Accuracy')
    plt.title('Logistic Regression: Accuracy vs C Hyperparameter')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Train final model with best C
    best_model = LogisticRegression(C=best_C, max_iter=1000, random_state=42)
    best_model.fit(X_train, y_train)

    return best_model


def train_logistic_regression(X: np.ndarray, y: np.ndarray):
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(X, y)

    model = tune_hyperparameters(X_train, y_train, X_val, y_val)
    plot_learning_curve(model, X, y)
    plot_roc_curve(model, X, y)

    train_acc = model.score(X_train, y_train)
    val_acc = model.score(X_val, y_val)
    test_acc = model.score(X_test, y_test)

    print(f"Train accuracy: {train_acc * 100:.2f}%")
    print(f"Val accuracy:   {val_acc * 100:.2f}%")
    print(f"Test accuracy:  {test_acc * 100:.2f}%")

    precision, recall = evaluate_precision_and_recall(model, X_test, y_test)
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")

    return model