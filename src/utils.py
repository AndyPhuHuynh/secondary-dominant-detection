import numpy as np

from sklearn.model_selection import train_test_split

def split_dataset(X: np.ndarray, y: np.ndarray, val_size: int = 0.2, test_size: int = 0.2) -> \
        tuple[np.ndarray, np.ndarray, \
                np.ndarray, np.ndarray, \
                np.ndarray, np.ndarray]:
    """
    Splits dataset into training, validation, and test sets
    :param X: Features of the dataset
    :param y: Labels of the dataset
    :param val_size: The percentage of data to be split into the validation set
    :param test_size: The percentage of data to be split into the test set
    :return: (X_train, y_train, X_val, y_val, X_test, y_test): training, validation, and test sets
    """
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    val_ratio = (val_size * len(y)) / len(y_temp)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp
    )

    return X_train, y_train, X_val, y_val, X_test, y_test