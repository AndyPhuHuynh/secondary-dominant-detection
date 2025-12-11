import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import learning_curve

from src.utils import split_dataset


def plot_learning_curve(model, X_train, y_train, X_val, y_val):
    """Generate and plot learning curves for training and validation sets"""
    train_sizes = np.linspace(0.1, 1.0, 10)

    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, X_train, y_train,
        train_sizes=train_sizes,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes_abs, train_mean, 'o-', color='r', label='Training score')
    plt.plot(train_sizes_abs, val_mean, 'o-', color='g', label='Cross-validation score')

    plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std,
                     alpha=0.1, color='r')
    plt.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std,
                     alpha=0.1, color='g')

    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy Score')
    plt.title('Learning Curves')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def tune_hyperparameters(X_train, y_train, X_val, y_val):
    c_values = [0.001, 0.01, 0.1, 1, 10, 100]
    best_score = 0
    best_c = None
    best_model = None

    results = []

    for c in c_values:
        model = LogisticRegression(
            C=c,
            max_iter=1000,
            random_state=0
        )
        model.fit(X_train, y_train)

        train_score = model.score(X_train, y_train)
        val_score = model.score(X_val, y_val)

        results.append({
            "C": c,
            "train_acc": train_score,
            "val_acc": val_score
        })

        print(f"C={c:2.2f}, Train: {train_score*100:.2f}, Val: {val_score*100:.2f}")
        if val_score > best_score:
            best_score = val_score
            best_c = c
            best_model = model

    print(f"Best C value is: {best_c}")
    return best_model, best_c, results


def plot_validation_curve(results):
    """Plot training vs validation accuracy for different C values"""
    C_values = [r['C'] for r in results]
    train_accs = [r['train_acc'] for r in results]
    val_accs = [r['val_acc'] for r in results]

    plt.figure(figsize=(10, 6))
    plt.semilogx(C_values, train_accs, 'o-', label='Training accuracy', linewidth=2)
    plt.semilogx(C_values, val_accs, 's-', label='Validation accuracy', linewidth=2)
    plt.xlabel('C (Regularization parameter)')
    plt.ylabel('Accuracy')
    plt.title('Validation Curve: Model Performance vs Regularization')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def train_baseline(X: np.ndarray, y: np.ndarray):
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(X, y)

    model, c, results = tune_hyperparameters(X_train, y_train, X_val, y_val)
    plot_validation_curve(results)
    plot_learning_curve(model, X_train, y_train, X_val, y_val)

    train_acc = model.score(X_train, y_train)
    val_acc = model.score(X_val, y_val)
    test_acc = model.score(X_test, y_test)

    print(f"Train accuracy: {train_acc * 100:.2f}%")
    print(f"Val accuracy:   {val_acc * 100:.2f}%")
    print(f"Test accuracy:  {test_acc * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, model.predict(X_test)))
    print("\nClassification Report:")
    print(classification_report(y_test, model.predict(X_test),
                                target_names=['Diatonic', 'Non-diatonic']))

    return model