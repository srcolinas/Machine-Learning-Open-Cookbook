import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def load_toy_dataset(filepath, feature_names, label_name="y"):
    """Loads a toy dataset from its filepath.

    Args:
        filepath (str): path to the toy dataset
        feature_names (list or None): column names to use as features.
        label_name (str): name of the output variable, defaults to 'y'.
    
    Returns:
    (tuple): Numpy arrays corresponding to X_train, y_train, X_test, y_test

    """
    df = pd.read_csv(filepath)
    df_train = df[df["split"] == "train"]
    X_train, y_train = df_train[feature_names].values, df_train[label_name].values
    df_test = df[df["split"] == "test"]
    X_test, y_test = df_test[feature_names].values, df_test[label_name].values
    return X_train, y_train, X_test, y_test


def plot_univariate_regression(
    X_train, y_train, X_test=None, y_test=None, model_fn=None, preprocessing_fn=None
):

    _, ax = plt.subplots()

    xmin = np.min(X_train)
    xmax = np.max(X_train)

    if X_test is not None:
        xmin = min(np.min(X_test), xmin)
        xmax = max(np.max(X_test), xmax)
        ax.scatter(X_test[:, 0], y_test, label="Test", c="tab:green", alpha=0.6)

    if model_fn is not None:
        X_real = np.linspace(xmin, xmax, 1000000).reshape(-1, 1)
        if preprocessing_fn is not None:
            X_real = preprocessing_fn(X_real)
        y_pred = model_fn(X_real)
        ax.plot(X_real[:, 0], y_pred, label="Model", c="tab:orange")

    ax.scatter(X_train[:, 0], y_train, label="Train", c="tab:blue", alpha=0.6)

    ymin = y_train.min()
    ymax = y_train.max()
    if y_test is not None:
        ymin = min(ymin, y_test.min())
        ymax = max(ymax, y_test.max())

    ax.set_ylim(ymin, ymax)
    ax.legend()
    plt.show()


def plot_bivariate_classifcation(
    X_train, y_train, X_test=None, y_test=None, model_fn=None, preprocessing_fn=None
):

    _, ax = plt.subplots()

    cmap = ListedColormap(["#00aeff", "#ff8400"])
    # plot the decision surface
    x1_min, x1_max = X_train[:, 0].min() - 0.2, X_train[:, 0].max() + 0.2
    x2_min, x2_max = X_train[:, 1].min() - 0.2, X_train[:, 1].max() + 0.2

    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, 0.01), np.arange(x2_min, x2_max, 0.01)
    )

    if model_fn is not None:
        X_real = np.array([xx1.ravel(), xx2.ravel()]).T
        if preprocessing_fn is not None:
            X_real = preprocessing_fn(X_real)
        z = model_fn(X_real)
        z = np.reshape(z, xx1.shape)
        ax.contourf(xx1, xx2, z, alpha=0.1, cmap=cmap)

    ax.set_xlim(xx1.min(), xx1.max())
    ax.set_ylim(xx2.min(), xx2.max())

    ax.scatter(X_train[:, 0], X_train[:, 1], label='Train', c=y_train, cmap=cmap)
    if X_test is not None:
        ax.scatter(X_test[:, 0], X_test[:, 1], label='Test', marker='^', c=y_test, cmap=cmap)

    ax.legend()
    plt.show()
