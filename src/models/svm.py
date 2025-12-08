from sklearn.svm import SVC

from src.utils import split_dataset


def train_svm(X, y):
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(X, y)
    svm = SVC(kernel="rbf", C=1.0, gamma="scale")
    svm.fit(X_train, y_train)
    print(f"SVM Train:      ", svm.score(X_train, y_train))
    print(f"SVM Validation: ", svm.score(X_val, y_val))
    print(f"SVM Test:       ", svm.score(X_test, y_test))