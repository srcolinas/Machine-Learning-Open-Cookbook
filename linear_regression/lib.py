from collections import defaultdict

import numpy as np

def mean_squared_error(y_true, y_pred):
    """Computes the mean squared error between to arrays.

    Args:
        y_true (numpy.ndarray) : array of true values of shape (n_samples, ).
        y_pred (numpy.ndarray) : array of predicted values of shape
            (n_samples, ).

    Returns:
        float : the mean squared errors between `y_true` and `y_pred`.
    """
    return np.mean((y_true - y_pred)**2)

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
        
    def fit(self, X, y, method='normal equations', learning_rate=0.001,
        num_iterations=100, tolerance=1e-10, log_every_n_steps=0.1,
        log_weights=True):
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

        X = self._maybe_add_ones(X)
        y = y[:, None]
        if method == 'normal equations':
            self._fit_by_normal_equations(X, y)
        elif method == 'gradient descent':
            self._fit_by_gradient_descent(X, y, learning_rate, num_iterations,
                tolerance, log_every_n_steps, log_weights)
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
        self._coef = np.reshape(weights, (-1, ))

        
    def _fit_by_gradient_descent(self, X, y, learning_rate, num_iterations,
        tolerance, log_every_n_steps, log_weights):
        """Fits a linear regression model using gradient descent"""
        if isinstance(log_every_n_steps, float):
            log_every_n_steps = int(log_every_n_steps * num_iterations)

        logging = defaultdict(list)
        num_variables = X.shape[1]
        w = np.random.randn(num_variables, 1)
        prev_loss = np.inf
        for i in range(num_iterations):
            dw = self._compute_gradient(X, y, w)
            w = w - learning_rate * dw
            
            if i % log_every_n_steps == 0:
                y_pred = self._compute_predictions(X, w=w)
                loss = mean_squared_error(y, y_pred)
                logging['loss'].append(loss)
                if log_weights:
                    logging['weights'].append(w)

                if abs(loss - prev_loss) <= tolerance:
                    break
            
        self._coef = np.reshape(w, (-1, ))
        logging = {k: tuple(values) for k, values in logging.items()}
        self._learning_curves = logging
       
    
    def _compute_gradient(self, X, y, w=None, y_pred=None):
        """Computes the gradient of the mean mquared error loss function.
        
        This method can take either `w` or `y_pred` to compute the gradient.

        Args:
            X (numpy.ndarray) : array of shape (n_samples, n_features).
            y (numpy.ndarray) : array of true values of shape (n_samples, )
            w (numpy.ndarray) : array of shape (n_features, ). Defaults to None.
            y_pred (numpy.ndarray) : array of predicted values of shape
                (n_samples, )

        Returns:
            The gradient of the mean squeared error loss function with
            respect to w.

        """
        if y_pred is None:
            y_pred = self._compute_predictions(X, w)
        diff = y_pred - y
        return np.mean(X * diff, axis=0)[..., None]
        
    def _compute_predictions(self, X, w):
        """Comutes a linear regression prediction.
        
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
        return np.dot(X, w)
            
    def predict(self, X):
        """Make predictions for a given input X.
        
        The model must first be fitted or a RunTimeException will be rised.

        Args:
            X (numpy.ndarray) : array of shape (n_samples, n_features)
        
        Returns:
            (numpy.ndarray) : predictions made with trained coeficients. 
        """ 
        X = self._maybe_add_ones(X)
        pred = self._compute_predictions(X, w=self.coef[:, None])[:, 0]
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