import numpy as np


def sigmoid(z):
    """Computes the sigmoid activation function of z. """
    a = 1/(1+np.exp(-z))
    return a

class LogisticRegression:
    def __init__(self):
        pass

    def _maybe_add_ones(self, X):
        """Concatenates at right a column of ones to 2d array X. """
        if self._include_bias:        
            n_rows, _ = X.shape
            ones = np.ones((n_rows, 1))
            X = np.concatenate((ones, X), axis=-1)
        return X

    def fit(self, X, y, learning_rate=0.001, num_iterations=100,
        tolerance=1e-10, log_every_n_steps=0.1, log_weights=True):
        """Fits a logistic regression model using the gradient descent method.

        Args:
            X (numpy.ndarray): Array of predictor variables of shape [n_samples, n_features].
            y (numpy.ndarray): Target values of shape [n_samples,].
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

    def predict(self, X, return_probs=False):
        pass

    def _compute_predictions(self, X, w, return_probs=False):
        probs = sigmoid(np.dot(X, w)[:, 0])
        if return_probs:
            return probs
        else:
            return (probs >= self.threshold)

    def _compute_cost(self, y, y_pred):
        cost = -np.mean(y*np.log(y_pred) + (1-y)*np.log(1-y_pred))
        return cost

    def _compute_gradient((self, X, y, w=None, y_pred=None):
        """Computes the gradient of the cross entropy loss function.
        
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
            y_pred = self._compute_predictions(X, w, return_probs=True)
        grad = np.dot(X.T, y_pred - y)/X.shape[0]
        return grad

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
        probs = sigmoid(np.dot(X, w)[:, 0])
        if return_probs:
            return probs
        else:
            return (probs >= self.threshold).astype(int)

    def predict(self, X):
        """Make predictions for a given input X.
        
        The model must first be fitted or a RunTimeException will be rised.

        Args:
            X (numpy.ndarray) : array of shape (n_samples, n_features)
        
        Returns:
            (numpy.ndarray) : predictions made with trained coeficients. 
        """ 
        X = self._maybe_add_ones(X)
        pred = self._compute_predictions(X, w=self.coef[:, None])
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
