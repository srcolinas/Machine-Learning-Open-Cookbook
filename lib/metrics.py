import numpy as np

def mean_squared_error(y_true, y_pred):
    """Computes the mean squared error between to arrays.

    mse = np.mean((y_true - y_pred)**2)
    Args:
        y_true (numpy.ndarray) : array of true values of shape (n_samples, ).
        y_pred (numpy.ndarray) : array of predicted values of shape
            (n_samples, ).

    Returns:
        float : the mean squared errors between `y_true` and `y_pred`.
    """
    return np.mean((y_true - y_pred)**2)