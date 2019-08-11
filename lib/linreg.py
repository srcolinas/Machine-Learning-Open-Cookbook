"""In this module we implement basic linear regression techniques. """

from collections import defaultdict

import numpy as np

from lib.metrics import mean_squared_error

class LinearRegression:
    """This class implements a linear regression model.
    
    Methods:
        fit: find the parameters of the linear regresion model.
        predict: makes a prediction for a new input X.
        
    Properties:
        coef (numpy.array): the coeficients of the linear regression model.
            Note that this can only be accesed after the `fit` method has been
            called.
        learning_curves (list): the learning curves of the training process.
            Note that this only make sense if you fit the model using the
            'gradient descent' method.
        
    """
    def __init__(self, include_bias=True):
        self._include_bias = include_bias

        self._coef = None
        self._learning_curves = None
        
    def fit(self, X, y, method, **kwargs):
        """Train a linear regression models
        
        Args:
            X (numpy.ndarray): Predictor variables values
            y (numpy.ndarray): Target values
            method (str): either 'normal equations' or 'grandient descent'. If
                you fit the model with 'gradient descent' you can access the
                learning curves of the training process.
                
        Returns:
            Returns `self` for convenient API.

        """

        if self._include_bias:
            X = self._add_ones(X)
        
        if method == 'normal equations':
            self._fit_by_normal_equations(X, y)
        elif method == 'gradient descent':
            self._fit_by_gradient_descent(X, y)
        else:
            msg = "method must be one of 'normal equations' or 'gradent descent'"
            raise ValueError(msg)
        
        return self
     
    def _add_ones(self, X):
        """Concatenates at right a column of ones to 2d array X. """
        n_rows, _ = X.shape
        ones = np.ones((n_rows, 1))
        X_ = np.concatenate((ones, X), axis=-1)
        return X_
    
    def _fit_by_normal_equations(self, X, y):
        """Fits a linear regression model using normal equations."""
        XTX = np.dot(X.T, X)
        XTX_inv = np.linalg.inv(XTX)
        XTy = np.dot(X.T, y)
        weights = np.dot(XTX_inv, XTy)
        self._coef = np.reshape(weights, (-1, ))

        
    def _fit_by_gradient_descent(self, X, y, **kwargs):
        """Fits a linear regression model using gradient descent"""
        learning_rate = kwargs.get('learning_rate', 0.01)
        num_iterations = kwargs.get('num_iterations', 10)
        
        logging = defaultdict(list)
        log_every_n_steps = kwargs.get('log_every_n_steps', num_iterations)
        log_loss = kwargs.get('log_loss', True)
        log_weights = kwargs.get('log_weights', True)
        
        num_variables = X.shape[1]
        w = np.random.randn(num_variables, 1)
        
        for i in range(num_iterations):
            dw = self._compute_gradient(X, y, w)
            w = w - learning_rate * dw
            
            if i % log_every_n_steps == 0:
                if log_loss:
                    y_pred = self._compute_predictions(X, w=w)
                    loss = mean_squared_error(y[:, 0], y_pred)
                    logging['loss'].append(loss)
                if log_weights:
                    logging['weights'].append(w)
            
        self._coef = w
        logging = {k: tuple(values) for k, values in logging.items()}
        self._learning_curves = logging
       
    
    def _compute_gradient(self, X, y, w):
        """Computes the gradient of the mean mquared error loss function.
        
        Args:
            X (numpy.ndarray) : array of shape (n_samples, n_features).
            y (numpy.ndarray) : array of true values of shape (n_samples, )
            w (numpy.ndarray) : array of shape (n_features, )

        Returns:
            The gradient of the mean squeared error loss function with
            respect to w.

        """
        pred = self._compute_predictions(X, w=w)
        diff = pred - y
        return np.mean(X * diff, axis=0)[..., None]
        
    def _compute_predictions(self, X, w):
        """Comutes a linear regression prediction.
        
        The prediction is done by the following equation for each of the
        provided samples in X: 
            `y = x0*w0 + x1*w1 + ... + xd*wd`

        Args:
            X (numpy.ndarray) : array of shape (n_samples, n_features).
            w (numpy.ndarray) : array of shape (n_features, )

        Returns:
            (numpy.ndarray) : array of shape (n_samples, 1) corresponding to
                a prediction for each of the samples in X.
        """
        return np.dot(X, w[:, None])
            
    def predict(self, X):
        """Make predictions for a given input X.
        
        The model must first be fitted or a RunTimeException will be rised.

        Args:
            X (numpy.ndarray) : array of shape (n_samples, n_features)
        
        Returns:
            (numpy.ndarray) : predictions made with trained coeficients. 
        """ 
        pred = self._compute_predictions(X, w=self.coef)[:, 0]
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
    pass