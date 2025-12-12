import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC

from src.features.utils import evaluate_precision_and_recall
from src.visualization.learning_curve import plot_learning_curve
from src.visualization.roc import plot_roc_curve
from src.utils import split_dataset

def tune_svm(X, y):
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(X, y)
    c_values = np.logspace(-3, 3, 7)
    gamma_values = np.logspace(-4, 0, 5)

    val_scores = np.zeros((len(c_values), len(gamma_values)))
    for i, C in enumerate(c_values):
        for j, gamma in enumerate(gamma_values):
            svm = SVC(kernel="rbf", C=C, gamma=gamma)
            svm.fit(X_train, y_train)
            val_scores[i, j] = svm.score(X_val, y_val)

    return c_values, gamma_values, val_scores


def plot_svm_heatmap(C_values, gamma_values, val_scores):
    plt.figure(figsize=(8, 6))
    plt.imshow(val_scores, origin="lower", aspect="auto")
    plt.xticks(range(len(gamma_values)), gamma_values)
    plt.yticks(range(len(C_values)), C_values)

    plt.xlabel("Gamma")
    plt.ylabel("C")
    plt.title("SVM Hyperparameter Tuning Heatmap")
    plt.colorbar(label="Validation Accuracy")
    plt.show()


def get_best_hyperparameters(c_values, gamma_values, val_scores):
    max_index = np.unravel_index(np.argmax(val_scores, axis=None), val_scores.shape)
    best_C = c_values[max_index[0]]
    best_gamma = gamma_values[max_index[1]]
    return best_C, best_gamma


def train_svm(X, y):
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(X, y)
    c_values, gamma_values, val_scores = tune_svm(X, y)
    plot_svm_heatmap(c_values, gamma_values, val_scores)
    best_c, best_gamma = get_best_hyperparameters(c_values, gamma_values, val_scores)
    print(f"Best SVM Hyperparameters: C={best_c}, gamma={best_gamma}")

    svm = SVC(kernel="rbf", C=best_c, gamma=best_gamma)
    plot_learning_curve(svm, X, y)
    plot_roc_curve(svm, X, y)

    svm.fit(X_train, y_train)
    print(f"SVM Train:      ", svm.score(X_train, y_train))
    print(f"SVM Validation: ", svm.score(X_val, y_val))
    print(f"SVM Test:       ", svm.score(X_test, y_test))

    precision, recall = evaluate_precision_and_recall(svm, X_test, y_test)
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")