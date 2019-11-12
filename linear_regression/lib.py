from collections import defaultdict

import numpy as np

def linear_prediction(X, w):
    """Comutes a linear prediction.
    
    The prediction is done by the following equation for each of the
    provided samples in X: 
        `y = x0*w0 + x1*w1 + ... + xd*wd`

    Args:
        X (numpy.ndarray) : array of shape (n_samples, n_features).
        w (numpy.ndarray) : array of shape (n_features, 1)

    Returns:
        (numpy.ndarray) : array of shape (n_samples, 1) corresponding to
            a prediction for each of the samples in X.
    """
    
    preds = np.dot(X, w)
    return preds

def mean_squared_error(y_true, y_pred):
    """Computes the mean squared error between to arrays.

    Args:
        y_true (numpy.ndarray) : array of true values of shape (n_samples, ).
        y_pred (numpy.ndarray) : array of predicted values of shape
            (n_samples, ).

    Returns:
        float : the mean squared error between `y_true` and `y_pred`.
    """
    return np.mean((y_true - y_pred) ** 2)

def mean_squared_error_gradient(X, y_true, y_pred):
    """Computes the gradient of the mean mquared error loss function.

    This method ()can take either `w` or `y_pred` to compute the gradient.

    Args:
        X (numpy.ndarray) : array of shape (n_samples, n_features).
        y_true (numpy.ndarray) : array of true values of shape (n_samples, ) or (n_samples, 1).
        y_pred (numpy.ndarray) : array of predicted values of shape (n_samples, ) or (n_samples, 1).

    Returns:
        The gradient of the mean squeared error loss function with
        respect to w.

    """
    grad = np.dot(-1 * X.T, y_true - y_pred)
    return grad

def l2_regularized_mean_squared_error(y_true, y_pred, w, reg):
    return mean_squared_error(y_true, y_pred) + reg * np.linalg.norm(w)

def l2_regularized_mean_squared_error_gradient(X, y_true, y_pred, w, reg):
    """
    Computes the gradient of the mean mquared error loss function with
    L2 regularization.
    
    This method can take either `w` or `y_pred` to compute the gradient.

    Args:
        X (numpy.ndarray) : array of shape (n_samples, n_features).
        y_true (numpy.ndarray) : array of true values of shape (n_samples, 1)
        y_pred (numpy.ndarray) : array of predicted values of shape
            (n_samples, 1)
        w (numpy.ndarray) : array of shape (n_features, 1).
        reg (float) : the regularization coefficient.

    Returns:
        The gradient of the mean squeared error loss function with L2
        regularization with respect to w.

    """
    grad = np.dot(-1 * X.T, y_true - y_pred) + reg*w
    return grad


class LinearRegression:
    """This class implements a linear regression model.
    
    Methods:
        fit: find the parameters of the linear regresion model.
        predict: makes a prediction for a new input X.
        
    Properties:
        coef (numpy.array): the coeficients of the linear regression model.
            Note that this can only be accesed after the `fit` method has been
            called.
        learning_curves (dict): the learning curves of the training process.
            Note that this only make sense if you fit the model using the
            'gradient descent' method.
        
    """

    def __init__(self, include_bias=True):
        self._include_bias = include_bias

        self._coef = None
        self._learning_curves = None

    def fit(
        self,
        X,
        y,
        method="normal equations",
        learning_rate=0.001,
        num_iterations=100,
        tolerance=1e-10,
        log_every_n_steps=0.1,
        log_weights=True,
    ):
        """Train a linear regression model.
        
        Args:
            X (numpy.ndarray): Predictor variables values
            y (numpy.ndarray): Target values
            method (str): either 'normal equations' or 'grandient descent'. If
                you fit the model with 'gradient descent' you can access the
                learning curves of the training process. Defaults to 'normal equations'.
            learning_rate (float): learning rate to use if using the gradient
                descent method. Defaults to 0.001.
            num_iterations (int): number of iterations for which to run gradient
                descent. Defaults to 100.
            tolerance (float): If using gradient descent, the algorithm will be
                stoped if successive iterations do not decrease the mean squared
                error bellow this value.
            log_every_n_steps (float or int): When using the gradient descent
                method, this value controls how often to log the loss function
                and possibly the weights (if `log_weights` is set to `True`)
                of the model. If float then it represents a fraction of the
                `num_iterations` argument.
            log_weights (bool): whether to log the weights when using gradient
                descent method, which could be used to debug the learning
                algorithm. Defaults to `True`.
        """

        X = self._maybe_add_ones(X.copy())
        y = y[:, None]
        if method == "normal equations":
            self._fit_by_normal_equations(X, y)
        elif method == "gradient descent":
            self._fit_by_gradient_descent(
                X,
                y,
                learning_rate,
                num_iterations,
                tolerance,
                log_every_n_steps,
                log_weights,
            )
        else:
            msg = "method must be one of 'normal equations' or 'gradent descent'"
            raise ValueError(msg)

    def _maybe_add_ones(self, X):
        """Concatenates at right a column of ones to 2d array X. """
        if self._include_bias:
            n_rows, _ = X.shape
            ones = np.ones((n_rows, 1))
            X = np.concatenate((ones, X), axis=-1)
        return X

    def _fit_by_normal_equations(self, X, y):
        """Fits a linear regression model using normal equations."""
        XTX = np.dot(X.T, X)
        XTX_inv = np.linalg.inv(XTX)
        XTy = np.dot(X.T, y)
        weights = np.dot(XTX_inv, XTy)
        self._coef = weights

    def _fit_by_gradient_descent(
        self,
        X,
        y,
        learning_rate,
        num_iterations,
        tolerance,
        log_every_n_steps,
        log_weights,
    ):
        """Fits a linear regression model using gradient descent"""
        if isinstance(log_every_n_steps, float):
            log_every_n_steps = int(log_every_n_steps * num_iterations)

        logging = defaultdict(list)
        num_variables = X.shape[1]
        self._w = np.random.randn(num_variables, 1)
        loss = self._compute_loss(y, linear_prediction(X, self._w))
        logging["loss"].append(loss)
        if log_weights:
            logging["weights"].append(self._w)

        prev_loss = loss
        for i in range(num_iterations):
            dw = self._compute_gradient(X, y)
            self._w = self._w - learning_rate * dw

            if i % log_every_n_steps == 0:
                y_pred = linear_prediction(X, self._w)
                loss = self._compute_loss(y, y_pred)
                logging["loss"].append(loss)
                if log_weights:
                    logging["weights"].append(self._w)

                if abs(loss - prev_loss) <= tolerance:
                    break

        self._coef = self._w
        logging = {k: tuple(values) for k, values in logging.items()}
        self._learning_curves = logging

    def _compute_loss(self, y_true, y_pred):
        return mean_squared_error(y_true, y_pred)

    def _compute_gradient(self, X, y_true):
        return mean_squared_error_gradient(X, y_true, linear_prediction(X, self._w))

    def predict(self, X):
        """Make predictions for a given input X.
        
        The model must first be fitted or a RunTimeException will be rised.

        Args:
            X (numpy.ndarray) : array of shape (n_samples, n_features)
        
        Returns:
            (numpy.ndarray) : predictions made with trained coeficients. 
        """
        X = self._maybe_add_ones(X)
        pred = linear_prediction(X, self.coef)[:, 0]
        return pred

    @property
    def coef(self):
        if self._coef is None:
            raise RuntimeError("You must fit the model first")
        return self._coef

    @property
    def learning_curves(self):
        if self._learning_curves is None:
            msg = "You must first fit the model with the gradient descent method"
            raise RuntimeError(msg)
        return self._learning_curves


class RidgeRegression(LinearRegression):
    def __init__(self, reg, include_bias=True):
        super().__init__(include_bias)
        self._reg = reg

    def _fit_by_normal_equations(self, X, y):
        """Fits a linear regression model using normal equations."""
        XTX = np.dot(X.T, X)
        XTX_inv = np.linalg.inv(XTX + self._reg*np.eye(X.shape[1]))
        XTy = np.dot(X.T, y)
        weights = np.dot(XTX_inv, XTy)
        self._coef = weights

    def _compute_loss(self, y_true, y_pred):
        return regularized_mean_squared_error(y_true, y_pred, self._w, self._reg)

    def _compute_gradient(self, X, y_true):
        y_pred = linear_prediction(X, self._w)
        return regularized_mean_squared_error_gradient(X, y_true, y_pred, self._w, self._reg)   